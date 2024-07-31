#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
//#include "kernel_col.h"
#include "exo_matrix.h"

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

void simplegemm(int M, int N, int K, const float * A, const float * B, float *C);
void initialize(int M, int N, int K, float * A, float *B, float *C, float *Ce);

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
  ukrFunction*** ukrmatrix = allocateMatrix();
  fillMatrix(ukrmatrix,beta0);
  float alpha = 1.0;
  float beta = 1.0;
  double GF[25][25] = {{0.0}};
  float * A = malloc(sizeof(float)*Mf*K);
  float * B = malloc(sizeof(float)*Nf*K);
  float * C = malloc(sizeof(float)*Mf*Nf);
  float * Ce = malloc(sizeof(float)*Mf*Nf);
  initialize(Mf,Nf,K, A, B, C, Ce);
  ukrFunction ukr_au = *ukrmatrix[Mi][Ni];
  ukr_au(NULL, K, &alpha, (struct exo_win_2f32c){A,{Mi,1}},(struct exo_win_2f32c) {B,{Ni,1}}, &beta, (struct exo_win_2f32){Ce,{Mi,1}});
  
  initialize(Mf,Nf,K, A, B, C, Ce);
  for (int ii =Mi; ii<=Mf; ii++){
	  for(int jj=Ni;jj<=Nf;jj++){
              int M = ii; int N = jj;
              initialize(M,N,K, A, B, C, Ce);
	      double gflops = (2.0*M*N*K)/1e9;

	      ukrFunction ukr = *ukrmatrix[M][N];
              if (ukr == NULL){
                 printf("Error! The desired ukernel does not exist!\n");
                return -1;
             }
	      //int NM= (int)(ceil(M/4.0))*4;
             //printf("M=%d and New M=%d\n",M,NM);
             start = dclock();
             for (int s = 0; s < reps; s++){
                 //ukr(NULL, K, &alpha, A,B, &beta, (struct exo_win_2f32){Ce,{M,1}});
                 ukr(NULL, K, &alpha, 
				 (struct exo_win_2f32c){A,{M,1}},
				 (struct exo_win_2f32c){B,{N,1}}, 
				 &beta, 
				 (struct exo_win_2f32){Ce,{M,1}});
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

void simplegemm(int M, int N, int K, const float * A, const float * B, float *C){
   int Alda = M, Clda =  M;
   int Blda = N;   
   int    i, j, p;
    
   for ( p=0; p<K; p++ )
	   for ( j=0; j<N; j++ )
		   for ( i=0; i<M; i++ ){
			   Cref(i,j) = Cref(i,j) + Aref(i,p) * Bref(j,p);
		   }
}

void initialize(int M, int N,int K,float * A, float *B, float *C, float *Ce) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = (float)rand()/(float)RAND_MAX; //i*j*0.1; //(rand())/RAND_MAX; //(i * K + j) * 0.1;//*0.1;//3.2;
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = (float)rand()/(float)RAND_MAX; //i*j*0.2; //(rand())/RAND_MAX; //(i * N + j)*0.2;//*0.2;
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
