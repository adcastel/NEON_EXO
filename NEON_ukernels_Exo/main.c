#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
//#include "kernel_col.h"
#ifdef fp32
#include "exo_matrix_NEON_fp32.h"
#define DTYPE float
#else
#include "exo_matrix_NEON_fp16.h"
#define DTYPE _Float16
#endif
#define Aref(a1,a2)  A[ (a2)*(Alda)+(a1) ]
#define Bref(a1,a2)  B[ (a2)*(Blda)+(a1) ]
#define Cref(a1,a2)  C[ (a2)*(Clda)+(a1) ]

double dclock()
{
	struct timeval  tv;
	// struct timezone tz;
	gettimeofday( &tv, NULL );

	return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}

void simplegemm(int M, int N, int K, const DTYPE * A, const DTYPE * B, DTYPE *C);
void initialize(int M, int N, int K, DTYPE * A, DTYPE *B, DTYPE *C, DTYPE *Ce);

int main(int argc, char * argv []) {
  double start, end;
  double msec;
  int Mi = atoi(argv[1]);
  int Mf = atoi(argv[2]);
  int Ni = atoi(argv[3]);
  int Nf = atoi(argv[4]);
  int K = atoi(argv[5]);
  int reps = atoi(argv[6]);
  int beta0 = 1;
  ukrFunction**** ukrmatrix = allocateMatrix();
  fillMatrix(ukrmatrix);
  DTYPE alpha = 1.0;
  DTYPE beta = 1.0;
  double GF[25][25] = {{0.0}};
  DTYPE * A = malloc(sizeof(DTYPE)*Mf*K);
  DTYPE * B = malloc(sizeof(DTYPE)*Nf*K);
  DTYPE * C = malloc(sizeof(DTYPE)*Mf*Nf);
  DTYPE * Ce = malloc(sizeof(DTYPE)*Mf*Nf);
  initialize(Mf,Nf,K, A, B, C, Ce);
  ukrFunction ukr_au = *ukrmatrix[Mi][Ni][0];
  ukr_au(NULL, K, &alpha, A,Mi,B,Ni, &beta,Ce,Mi);
  
  initialize(Mf,Nf,K, A, B, C, Ce);
  for (int ii =Mi; ii<=Mf; ii++){
	  for(int jj=Ni;jj<=Nf;jj++){
              int M = ii; int N = jj;
              initialize(M,N,K, A, B, C, Ce);
	      double gflops = (2.0*M*N*K)/1e9;

	      ukrFunction ukr = *ukrmatrix[M][N][0];
              if (ukr == NULL){
                 printf("Error! The desired ukernel does not exist!\n");
                return -1;
              }
	      //int NM= (int)(ceil(M/4.0))*4;
             //printf("M=%d and New M=%d\n",M,NM);
             start = dclock();
             for (int s = 0; s < reps; s++){
                 ukr(NULL, K, &alpha,
				 A,M,
				 B,N,
				 &beta,
				 Ce,M);
             }
             end = dclock();

            msec = (end - start) /reps;
            int error = 0;
	    if (reps == 1){
	        for (int s = 0; s < reps; s++){
                    simplegemm(M,N,K,A,B,C);
	         }
               
		for(int i = 0; i< M; i++){
                   for(int j = 0; j< N; j++){
	                if(fabs(C[j* M + i] - Ce[j*M+i]) < 0.00001){
	  	             //printf("OK %dx%d %f %f abs(%f)\n",i,j,C[j*M+i],Ce[j*M+i],fabs(C[j* M + i] - Ce[j*M+i]));
		             continue;
		         }
	                else{
	  	             printf("ERROR %dx%d %f %f abs(%f)\n",i,j,C[j*M+i],Ce[j*M+i],fabs(C[j* M + i] - Ce[j*M+i]));
		             error = 1;
		             printf("E-");
			    break;
			}
                     }
	            if (error == 1){ error = 0; break;}
	           }
	    }

		
           printf("%d %d %d %f %f\n", M, N, K, msec, gflops/(msec)); fflush(stdout);
	   GF[M][N] = gflops/msec;
	  }
      }
      free(A); free(B); free(C); free(Ce);
      free(ukrmatrix);
      printf("## ");
      for(int j=Ni; j<=Nf; j++){
	  printf("%3d ", j);
       }
       printf("\n");
       for (int i=Mi; i<=Mf; i++){
           printf("%d ", i);
           for(int j=Ni; j<=Nf; j++){
 	      printf("%.2f ", GF[i][j]);
           }
           printf("\n");
       }
  //printf("PASS!\n");
  return (0);
}

void simplegemm(int M, int N, int K, const DTYPE * A, const DTYPE * B, DTYPE *C){
   int Alda = M, Clda =  M;
   int Blda = N;   
   int    i, j, p;
    
   for ( p=0; p<K; p++ )
	   for ( j=0; j<N; j++ )
		   for ( i=0; i<M; i++ ){
			   Cref(i,j) = Cref(i,j) + Aref(i,p) * Bref(j,p);
		   }
}

void initialize(int M, int N,int K,DTYPE * A, DTYPE *B, DTYPE *C, DTYPE *Ce) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = (DTYPE)rand()/(DTYPE)RAND_MAX; //i*j*0.1; //(rand())/RAND_MAX; //(i * K + j) * 0.1;//*0.1;//3.2;
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = (DTYPE)rand()/(DTYPE)RAND_MAX; //i*j*0.2; //(rand())/RAND_MAX; //(i * N + j)*0.2;//*0.2;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0.0;
      Ce[i * N + j] = 0.0;
    }
  }
  return;
}
