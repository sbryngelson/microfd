// microfd: 3D compressible Navier-Stokes on a uniform grid. Finite volume, WENO5-Z + HLLC, SSP-RK3.
// One file. OpenMP target offload (NVIDIA or AMD). MPI Cartesian decomposition with GPU-aware halos.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define NV 5                                   // rho, rho u, rho v, rho w, E
#define NG 3                                   // ghost layers (WENO5 stencil)
#define PRAGMA(x) _Pragma(#x)
#define LOCALS const int nx=g.n[0],ny=g.n[1],nz=g.n[2]; const long sx=g.sx,sy=g.sy; const size_t nc=g.nc; (void)(nx+ny+nz+sx+sy+nc)
#define IDX(i,j,k) ((i)+sx*(j)+sy*(k))
#define FOR3(i0,j0,k0,clauses) PRAGMA(omp target teams loop collapse(3) clauses) \
  for(int k=k0;k<nz+NG;k++) for(int j=j0;j<ny+NG;j++) for(int i=i0;i<nx+NG;i++)

static struct {                                // all solver state; kernels copy scalars to locals before offloading
  int N[3], n[3], e[3], bc[3], dims[3], coords[3], nb[3][2], rank, ndiag, nout, axis;
  long sx, sy; size_t nc, nbuf;                // strides (x stride is 1), padded cell count, halo buffer length
  double L[3], o[3], h[3], gamma, mu, pr, cfl, tend, t;
  double *q, *q1, *w, *F, *sbuf[2], *rbuf[2];      // F holds all three directions: [d][NV][nc]
  MPI_Comm comm;
} g;

static void die(const char*m){ fprintf(stderr,"microfd: %s\n",m); MPI_Abort(MPI_COMM_WORLD,1); }

// ---- cases: primitives (rho,u,v,w,p) at a point; the table gives domain, viscosity, end time, BCs (0 periodic, 1 wall, 2 outflow)
typedef struct { const char*name; void(*ic)(double,double,double,double*); double o[3],L[3],mu,tend; int bc[3]; } Case;
static double p0(double ma){ return 1/(g.gamma*ma*ma); }                                   // pressure giving Mach ma for rho=V=1
static void tgv(double x,double y,double z,double*p){ p[0]=1; p[1]=sin(x)*cos(y)*cos(z); p[2]=-cos(x)*sin(y)*cos(z); p[3]=0; p[4]=p0(.1)+(cos(2*x)+cos(2*y))*(cos(2*z)+2)/16; }
static void acoustic(double x,double y,double z,double*p){ const double e=1e-3*sin(x); p[0]=1+e; p[1]=sqrt(g.gamma)*e; p[2]=p[3]=0; p[4]=1+g.gamma*e; (void)y; (void)z; }
static void tgv2d(double x,double y,double z,double*p){ p[0]=1; p[1]=sin(x)*cos(y); p[2]=-cos(x)*sin(y); p[3]=0; p[4]=p0(.05)+(cos(2*x)+cos(2*y))/4; (void)z; }
static void sod(double x,double y,double z,double*p){ int l=(g.axis==0?x:g.axis==1?y:z)<.5; p[0]=l?1:.125; p[1]=p[2]=p[3]=0; p[4]=l?1:.1; }
static void sedov(double x,double y,double z,double*p){ p[0]=1; p[1]=p[2]=p[3]=0; p[4]=x*x+y*y+z*z<.01?(g.gamma-1)/4.18879e-3:1e-5; }  // E=1 in r<0.1
static void vortex(double x,double y,double z,double*p){ double b=5,dx=x-5,dy=y-5,f=exp(1-dx*dx-dy*dy),T=1-(g.gamma-1)*b*b*f/(8*g.gamma*M_PI*M_PI);
  p[0]=pow(T,1/(g.gamma-1)); p[1]=1-b*dy*sqrt(f)/(2*M_PI); p[2]=1+b*dx*sqrt(f)/(2*M_PI); p[3]=0; p[4]=p[0]*T; (void)z; }
static const Case cases[]={
  {"tgv",   tgv,   {-M_PI,-M_PI,-M_PI},{2*M_PI,2*M_PI,2*M_PI},1./1600,10,   {0,0,0}},
  {"acoustic",acoustic,{0,0,0},        {2*M_PI,2*M_PI,2*M_PI},.05,    4,    {0,0,0}},
  {"tgv2d", tgv2d, {0,0,0},            {2*M_PI,2*M_PI,2*M_PI},.1,     1,    {0,0,0}},
  {"sod",   sod,   {0,0,0},            {1,1,1},               0,         .2, {2,2,2}},
  {"sedov", sedov, {-1.2,-1.2,-1.2},   {2.4,2.4,2.4},         0,         .1, {2,2,2}},
  {"vortex",vortex,{0,0,0},            {10,10,10},            0,         10,   {0,0,0}}};

// ---- device helpers: reconstruction to the right face of cell c from the 5-cell stencil a..e, and Riemann solvers in the face-normal frame
#pragma omp declare target
static double weno5(double a,double b,double c,double d,double e){                 // WENO5-Z
  double b0=13./12*(a-2*b+c)*(a-2*b+c)+.25*(a-4*b+3*c)*(a-4*b+3*c);
  double b1=13./12*(b-2*c+d)*(b-2*c+d)+.25*(b-d)*(b-d);
  double b2=13./12*(c-2*d+e)*(c-2*d+e)+.25*(3*c-4*d+e)*(3*c-4*d+e);
  double t=fabs(b0-b2), w0=.1*(1+t/(b0+1e-16)), w1=.6*(1+t/(b1+1e-16)), w2=.3*(1+t/(b2+1e-16));
  return (w0*(2*a-7*b+11*c)+w1*(-b+5*c+2*d)+w2*(2*c+5*d-e))/(6*(w0+w1+w2));
}
static void hllc(const double*L,const double*R,double gam,double*f){           // HLLC with Davis wave speeds
  double rl=L[0],ul=L[1],pl=L[4],rr=R[0],ur=R[1],pr=R[4], cl=sqrt(gam*pl/rl),cr=sqrt(gam*pr/rr);
  double sl=fmin(ul-cl,ur-cr), sr=fmax(ul+cl,ur+cr), sm=(pr-pl+rl*ul*(sl-ul)-rr*ur*(sr-ur))/(rl*(sl-ul)-rr*(sr-ur));
  int left=sm>=0; const double*S=left?L:R; double s=left?sl:sr, r=S[0],u=S[1],p=S[4];
  double U[5]={r,r*u,r*S[2],r*S[3],p/(gam-1)+.5*r*(u*u+S[2]*S[2]+S[3]*S[3])};             // conserved state and flux of the upwind side
  for(int v=0;v<5;v++) f[v]=u*U[v]+(v==1)*p+(v==4)*p*u;
  if(left?sl<0:sr>0){ double k=(s-u)/(s-sm), Us[5]={r*k,r*k*sm,r*k*S[2],r*k*S[3],k*(U[4]+(sm-u)*(r*sm+p/(s-u)))};
    for(int v=0;v<5;v++) f[v]+=s*(Us[v]-U[v]); }
}
#pragma omp end declare target

// ---- kernels
static void prim(const double*q){                                        // conserved -> primitive over the whole padded block
  LOCALS; double*w=g.w; const double gm=g.gamma-1;
  #pragma omp target teams loop
  for(size_t c=0;c<nc;c++){ double r=q[c],u=q[nc+c]/r,v=q[2*nc+c]/r,s=q[3*nc+c]/r;
    w[c]=r; w[nc+c]=u; w[2*nc+c]=v; w[3*nc+c]=s; w[4*nc+c]=gm*(q[4*nc+c]-.5*r*(u*u+v*v+s*s)); }
}

static void face(int d){                                               // flux through the face c+1/2 normal to d, stored in F at cell c
  LOCALS; const double gam=g.gamma, mu=g.mu, kap=mu*gam/((gam-1)*g.pr), h0=g.h[0],h1=g.h[1],h2=g.h[2]; const double*w=g.w; double*F=g.F+(size_t)d*NV*nc;
  const int i0=NG-(d==0), j0=NG-(d==1), k0=NG-(d==2);
  FOR3(i0,j0,k0,){
    const long s=d==0?1:d==1?sx:sy, c=IDX(i,j,k); const int P[5]={0,1+d,1+(d+1)%3,1+(d+2)%3,4};   // face-normal frame
    double L[5],R[5],f[5];
    for(int v=0;v<5;v++){ const double*u=w+P[v]*nc+c; L[v]=weno5(u[-2*s],u[-s],u[0],u[s],u[2*s]); R[v]=weno5(u[3*s],u[2*s],u[s],u[0],u[-s]); }
    if(L[0]<=0||L[4]<=0) for(int v=0;v<5;v++) L[v]=w[P[v]*nc+c];        // positivity fallback: first order
    if(R[0]<=0||R[4]<=0) for(int v=0;v<5;v++) R[v]=w[P[v]*nc+c+s];
    hllc(L,R,gam,f);
    if(mu>0){                                                            // viscous stress and heat flux at the face, 2nd-order central
      const long st[3]={1,sx,sy}; const double h[3]={h0,h1,h2}; double du[3][3], div=0;
      for(int a=0;a<3;a++) for(int b=0;b<3;b++) if(a==b||a==d||b==d){ const double*u=w+(1+a)*nc+c; const long t=st[b];   // off-normal off-diagonal terms are dead
        du[a][b]= b==d ? (u[s]-u[0])/h[d] : (u[t]-u[-t]+u[s+t]-u[s-t])/(4*h[b]); }  // normal: two cells; tangential: averaged central
      for(int a=0;a<3;a++) div+=du[a][a];
      f[4]-=kap*(w[4*nc+c+s]/w[c+s]-w[4*nc+c]/w[c])/h[d];                              // heat flux with T = p/rho
      for(int m=0;m<3;m++){ const int a=(d+m)%3; const double tau=mu*(du[a][d]+du[d][a]-(a==d)*2./3*div);
        f[1+m]-=tau; f[4]-=tau*.5*(w[(1+a)*nc+c]+w[(1+a)*nc+c+s]); }
    }
    for(int v=0;v<5;v++) F[P[v]*nc+c]=f[v];
  }
}

static void update(double*out,double a,const double*qa,double b,const double*qb,double c){   // out = a qa + b qb - c div F; the divergence is fused in, so no rhs array
  LOCALS; const double*F=g.F; const double h0=g.h[0],h1=g.h[1],h2=g.h[2]; const size_t m=NV*nc;
  FOR3(NG,NG,NG,){ const long ci=IDX(i,j,k);
    for(int v=0;v<5;v++){ const size_t o=v*nc+ci;
      out[o]=a*qa[o]+b*qb[o]-c*((F[o]-F[o-1])/h0+(F[m+o]-F[m+o-sx])/h1+(F[2*m+o]-F[2*m+o-sy])/h2); } }
}

static double wavemax(void){                                             // max of sum_d (|u_d|+a)/h_d + 2 nu sum_d 1/h_d^2
  LOCALS; const double*q=g.q; const double gam=g.gamma,pr=g.pr,mu=g.mu,h0=g.h[0],h1=g.h[1],h2=g.h[2]; double m=0;
  FOR3(NG,NG,NG,reduction(max:m)){ const long c=IDX(i,j,k); double r=q[c],u=q[nc+c]/r,v=q[2*nc+c]/r,s=q[3*nc+c]/r;
    double a=sqrt(gam*(gam-1)*(q[4*nc+c]/r-.5*(u*u+v*v+s*s)));
    m=fmax(m,(fabs(u)+a)/h0+(fabs(v)+a)/h1+(fabs(s)+a)/h2+fmax(2,gam/pr)*mu/r*(1/(h0*h0)+1/(h1*h1)+1/(h2*h2))); }
  MPI_Allreduce(MPI_IN_PLACE,&m,1,MPI_DOUBLE,MPI_MAX,g.comm); return m;
}

// ---- halo exchange: x, then y, then z, so edges and corners arrive through the face slabs
enum {PACK,UNPACK,WALL,OUTFLOW};
static void slab(double*q,int d,int side,int mode,double*buf){             // NG-layer slab normal to d at side 0 (low) or 1 (high)
  LOCALS; const int e0=d==0?NG:g.e[0], e1=d==1?NG:g.e[1], e2=d==2?NG:g.e[2], N=g.n[d]; const size_t m=(size_t)NV*e0*e1*e2;
  #pragma omp target teams loop
  for(size_t t=0;t<m;t++){
    int x[3]={(int)(t%e0),(int)(t/e0%e1),(int)(t/e0/e1%e2)}; const int v=(int)(t/e0/e1/e2), l=x[d];
    const int gi=side?N+NG+l:NG-1-l, ii=side?N+NG-1-l:NG+l, bi=side?N+NG-1:NG;   // ghost cell, mirror cell, boundary cell along d
    x[d]=mode==PACK?ii:gi; const long c=v*nc+IDX(x[0],x[1],x[2]);
    if(mode==PACK) buf[t]=q[c]; else if(mode==UNPACK) q[c]=buf[t];
    else { x[d]=mode==WALL?ii:bi; q[c]=q[v*nc+IDX(x[0],x[1],x[2])]*(mode==WALL&&v==1+d?-1:1); }
  }
}
static void exchange(int d,size_t m,double*s0,double*s1,double*r0,double*r1){   // low slab goes down (tag 0), high slab goes up (tag 1)
  MPI_Request r[4];
  MPI_Irecv(r1,m,MPI_DOUBLE,g.nb[d][1],0,g.comm,r); MPI_Irecv(r0,m,MPI_DOUBLE,g.nb[d][0],1,g.comm,r+1);
  MPI_Isend(s0,m,MPI_DOUBLE,g.nb[d][0],0,g.comm,r+2); MPI_Isend(s1,m,MPI_DOUBLE,g.nb[d][1],1,g.comm,r+3);
  MPI_Waitall(4,r,MPI_STATUSES_IGNORE);
}
static void halo(double*q){
  double *s0=g.sbuf[0],*s1=g.sbuf[1],*r0=g.rbuf[0],*r1=g.rbuf[1];
  for(int d=0;d<3;d++){ const size_t m=NV*NG*(g.nc/g.e[d]);
    slab(q,d,0,PACK,s0); slab(q,d,1,PACK,s1);
#ifdef HOST_MPI
    #pragma omp target update from(s0[0:m],s1[0:m])
    exchange(d,m,s0,s1,r0,r1);
    #pragma omp target update to(r0[0:m],r1[0:m])
#else
    #pragma omp target data use_device_ptr(s0,s1,r0,r1)
    exchange(d,m,s0,s1,r0,r1);
#endif
    for(int s=0;s<2;s++) g.nb[d][s]==MPI_PROC_NULL ? slab(q,d,s,g.bc[d]==1?WALL:OUTFLOW,0) : slab(q,d,s,UNPACK,g.rbuf[s]);
  }
}
static void rhs_eval(double*q){ halo(q); prim(q); for(int d=0;d<3;d++) face(d); }

// ---- diagnostics and output
static void diag(int step,double dt,double wall){                        // mean kinetic energy and enstrophy, ns per cell per step
  LOCALS; const double*w=g.w; const double h0=g.h[0],h1=g.h[1],h2=g.h[2]; double ke=0,en=0;
  FOR3(NG,NG,NG,reduction(+:ke,en)){ const long c=IDX(i,j,k); const double r=w[c],*u=w+nc+c,*v=w+2*nc+c,*s=w+3*nc+c;
    double wx=(s[sx]-s[-sx])/(2*h1)-(v[sy]-v[-sy])/(2*h2), wy=(u[sy]-u[-sy])/(2*h2)-(s[1]-s[-1])/(2*h0), wz=(v[1]-v[-1])/(2*h0)-(u[sx]-u[-sx])/(2*h1);
    ke+=.5*r*(u[0]*u[0]+v[0]*v[0]+s[0]*s[0]); en+=.5*r*(wx*wx+wy*wy+wz*wz); }
  double sum[2]={ke,en}, N=(double)g.N[0]*g.N[1]*g.N[2]; MPI_Allreduce(MPI_IN_PLACE,sum,2,MPI_DOUBLE,MPI_SUM,g.comm);
  if(!g.rank){ printf("%d %.6e %.3e %.10e %.10e %.2f\n",step,g.t,dt,sum[0]/N,sum[1]/N,wall*1e9/N); fflush(stdout); }
}

static void output(int step){                                          // primitives as one raw [5][NZ][NY][NX] file via MPI-IO, plus XDMF
  LOCALS; double*w=g.w; const size_t m=NV*nc; char fn[64],xn[64];
  #pragma omp target update from(w[0:m])
  int gs[4]={NV,g.N[2],g.N[1],g.N[0]}, ls[4]={NV,nz,ny,nx}, st[4]={0,g.coords[2]*nz,g.coords[1]*ny,g.coords[0]*nx}, ms[4]={NV,g.e[2],g.e[1],g.e[0]}, mo[4]={0,NG,NG,NG};
  MPI_Datatype ft,mt; MPI_File f;
  MPI_Type_create_subarray(4,gs,ls,st,MPI_ORDER_C,MPI_DOUBLE,&ft); MPI_Type_commit(&ft);
  MPI_Type_create_subarray(4,ms,ls,mo,MPI_ORDER_C,MPI_DOUBLE,&mt); MPI_Type_commit(&mt);
  snprintf(fn,64,"out_%06d.bin",step); MPI_File_open(g.comm,fn,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&f); MPI_File_set_size(f,0);
  MPI_File_set_view(f,0,MPI_DOUBLE,ft,"native",MPI_INFO_NULL); MPI_File_write_all(f,w,1,mt,MPI_STATUS_IGNORE); MPI_File_close(&f);
  MPI_Type_free(&ft); MPI_Type_free(&mt);
  if(g.rank) return;
  static const char*name[]={"rho","u","v","w","p"}; snprintf(xn,64,"out_%06d.xmf",step); FILE*x=fopen(xn,"w");
  fprintf(x,"<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\"><Domain><Grid GridType=\"Uniform\"><Time Value=\"%g\"/>\n"
    "<Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"%d %d %d\"/><Geometry GeometryType=\"ORIGIN_DXDYDZ\">"
    "<DataItem Dimensions=\"3\" Format=\"XML\">%g %g %g</DataItem><DataItem Dimensions=\"3\" Format=\"XML\">%g %g %g</DataItem></Geometry>\n",
    g.t,g.N[2]+1,g.N[1]+1,g.N[0]+1,g.o[2],g.o[1],g.o[0],g.h[2],g.h[1],g.h[0]);
  for(int v=0;v<NV;v++) fprintf(x,"<Attribute Name=\"%s\" Center=\"Cell\"><DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" Precision=\"%d\" Format=\"Binary\" Seek=\"%zu\">%s</DataItem></Attribute>\n",
    name[v],g.N[2],g.N[1],g.N[0],(int)sizeof(double),(size_t)v*g.N[0]*g.N[1]*g.N[2]*sizeof(double),fn);
  fprintf(x,"</Grid></Domain></Xdmf>\n"); fclose(x);
}

// ---- main: options, decomposition, allocation, initial condition, time loop
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);
  char cname[32]="tgv"; for(int a=1;a<argc;a++) sscanf(argv[a],"case=%31s",cname);
  const Case*cs=0; for(size_t i=0;i<sizeof cases/sizeof*cases;i++) if(!strcmp(cases[i].name,cname)) cs=cases+i;
  if(!cs) die("unknown case");
  g.gamma=1.4; g.pr=.71; g.cfl=.5; g.ndiag=10; g.mu=cs->mu; g.tend=cs->tend;
  for(int d=0;d<3;d++){ g.N[d]=64; g.o[d]=cs->o[d]; g.L[d]=cs->L[d]; g.bc[d]=cs->bc[d]; }
  struct { const char*k; double*r; int*i; } opt[]={
    {"nx",0,&g.N[0]},{"ny",0,&g.N[1]},{"nz",0,&g.N[2]},{"px",0,&g.dims[0]},{"py",0,&g.dims[1]},{"pz",0,&g.dims[2]},
    {"x0",&g.o[0],0},{"y0",&g.o[1],0},{"z0",&g.o[2],0},{"lx",&g.L[0],0},{"ly",&g.L[1],0},{"lz",&g.L[2],0},
    {"bcx",0,&g.bc[0]},{"bcy",0,&g.bc[1]},{"bcz",0,&g.bc[2]},{"mu",&g.mu,0},
    {"cfl",&g.cfl,0},{"tend",&g.tend,0},{"nout",0,&g.nout},{"ndiag",0,&g.ndiag},{"axis",0,&g.axis},{"case",0,0}};
  const size_t nopt=sizeof opt/sizeof*opt;
  for(int a=1;a<argc;a++){ char k[32]; double v=0; size_t i=0;
    if(sscanf(argv[a],"%31[^=]=%lf",k,&v)<1) die("bad argument");
    while(i<nopt&&strcmp(opt[i].k,k)) i++;
    if(i==nopt) die("unknown key"); if(opt[i].r) *opt[i].r=v; else if(opt[i].i) *opt[i].i=(int)v; }
  if(g.ndiag<1) g.ndiag=1;                    // after parsing: ndiag=0 would divide by zero in the step test

  int np,per[3],lr; MPI_Comm loc; MPI_Comm_size(MPI_COMM_WORLD,&np); MPI_Dims_create(np,3,g.dims);
  for(int d=0;d<3;d++){ if(g.bc[d]<0||g.bc[d]>2) die("bc must be 0, 1 or 2"); per[d]=g.bc[d]==0; }
  MPI_Cart_create(MPI_COMM_WORLD,3,g.dims,per,1,&g.comm); MPI_Comm_rank(g.comm,&g.rank); MPI_Cart_coords(g.comm,g.rank,3,g.coords);
  for(int d=0;d<3;d++){ MPI_Cart_shift(g.comm,d,1,&g.nb[d][0],&g.nb[d][1]);
    if(g.N[d]%g.dims[d]) die("grid not divisible by ranks"); g.n[d]=g.N[d]/g.dims[d]; if(g.n[d]<NG) die("need at least 3 cells per rank per direction");
    g.e[d]=g.n[d]+2*NG; g.h[d]=g.L[d]/g.N[d]; }
  g.sx=g.e[0]; g.sy=(long)g.e[0]*g.e[1]; g.nc=(size_t)g.sy*g.e[2];
  for(int d=0;d<3;d++){ size_t m=NV*NG*(g.nc/g.e[d]); if(m>g.nbuf) g.nbuf=m; }   // a slab is the padded block with extent NG along d
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&loc); MPI_Comm_rank(loc,&lr);
  if(omp_get_num_devices()) omp_set_default_device(lr%omp_get_num_devices());

  const size_t m=NV*g.nc; double**arr[]={&g.q,&g.q1,&g.w,&g.F,&g.sbuf[0],&g.sbuf[1],&g.rbuf[0],&g.rbuf[1]};
  for(int i=0;i<8;i++) if(!(*arr[i]=calloc(i<3?m:i==3?3*m:g.nbuf,sizeof(double)))) die("out of memory");
  { LOCALS; for(int k=0;k<g.e[2];k++) for(int j=0;j<g.e[1];j++) for(int i=0;i<g.e[0];i++){   // IC on the padded block, ghosts included
      double p[5],x[3]; const int id[3]={i,j,k}; const long c=IDX(i,j,k);
      for(int d=0;d<3;d++) x[d]=g.o[d]+g.h[d]*(g.coords[d]*g.n[d]+id[d]-NG+.5);
      cs->ic(x[0],x[1],x[2],p); g.q[c]=p[0]; for(int v=1;v<4;v++) g.q[v*nc+c]=p[0]*p[v];
      g.q[4*nc+c]=p[4]/(g.gamma-1)+.5*p[0]*(p[1]*p[1]+p[2]*p[2]+p[3]*p[3]); } }
  double *q=g.q,*q1=g.q1,*w=g.w,*F=g.F,*s0=g.sbuf[0],*s1=g.sbuf[1],*r0=g.rbuf[0],*r1=g.rbuf[1]; const size_t nb=g.nbuf, m3=3*m;
  #pragma omp target enter data map(to:q[0:m]) map(alloc:q1[0:m],w[0:m],F[0:m3],s0[0:nb],s1[0:nb],r0[0:nb],r1[0:nb])

  double dt=0; int step=0, ls=0; double tl=MPI_Wtime();
  for(;;){
    { const int dd=step%g.ndiag==0||g.t>=g.tend, oo=g.nout&&(step%g.nout==0||g.t>=g.tend); if(dd||oo){ halo(g.q); prim(g.q); } if(dd){ diag(step,dt,(MPI_Wtime()-tl)/(step>ls?step-ls:1)); tl=MPI_Wtime(); ls=step; } if(oo) output(step); }
    if(g.t>=g.tend) break;
    dt=g.cfl/wavemax(); if(!(dt>0)) die("non-finite time step"); if(g.t+dt>g.tend) dt=g.tend-g.t;
    rhs_eval(g.q);  update(g.q1,1,g.q,0,g.q,dt);                        // SSP-RK3, two registers
    rhs_eval(g.q1); update(g.q1,.75,g.q,.25,g.q1,.25*dt);
    rhs_eval(g.q1); update(g.q,1./3,g.q,2./3,g.q1,2./3*dt);
    g.t+=dt; step++;
  }
  MPI_Finalize(); return 0;
}
