#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


#define INF 10000000
#define V 10010

int vertexNum, edgeNum;
static int graphMap[V*V];
int *graphDist;
int B;

void input(char *inFileName);
void output(char *outFileName);

__global__ void cudaFW_phase1(int ith_round, int vertexNum, int *graph_dist, int B)
{
    extern __shared__ int dist[]; 

    int x = threadIdx.x;
    int y = threadIdx.y;
    int global_x = x + ith_round * B;//check if its x out of graphmap
    int global_y = y + ith_round * B;//check if its y out of graphmap

    //if k, q(coordinate in matrix graph_dist) smaller than vertexNum
    if (global_x < vertexNum && global_y < vertexNum)
        dist[x * B + y] = graph_dist[global_x * vertexNum + global_y];
    if (global_x >= vertexNum || global_y >= vertexNum)
        dist[x * B + y] = INF;
    __syncthreads();

    #pragma unroll
    for (int i=0; i<B; i++) {
        if (dist[x*B + y] > dist[x*B + i] + dist[i*B + y])
            dist[x*B + y] = dist[x*B + i] + dist[i*B + y];
        __syncthreads();
    }
    if (global_x < vertexNum && global_y < vertexNum)
        graph_dist[global_x * vertexNum + global_y] = dist[x * B + y];
    __syncthreads();
}

__global__ void cudaFW_phase2(int ith_round, int vertexNum, int *graph_dist, int B)
{
   if (blockIdx.x != ith_round) {// block not equal to pivot block
        extern __shared__ int share_space[];//two matrices
        int *first_matrix = &share_space[0];//pivot block
        int *second_matrix = &share_space[B * B];//self block

        int x = threadIdx.x;
        int y = threadIdx.y;
        int global_x = x + ith_round * B;//pivot x;
        int global_y = y + ith_round * B;//pivot y;

        //assign pivot block to first_matrix
        if (global_x < vertexNum && global_y < vertexNum)
            first_matrix[x*B + y] = graph_dist[global_x * vertexNum + global_y];
        if (global_x >= vertexNum || global_y >= vertexNum)
            first_matrix[x*B + y] = INF;
        

        if (blockIdx.y == 0)
            global_x = x + blockIdx.x * B;//row blocks, self coordinate
        if (blockIdx.y != 0)
            global_y = y + blockIdx.x * B;//column blocks, self coordinate

        if (global_x < vertexNum && global_y < vertexNum) {
            second_matrix[x*B + y] = graph_dist[global_x*vertexNum + global_y];
            __syncthreads();

            if (blockIdx.y == 0) {//row block
                #pragma unroll
                for (int i=0; i<B; i++) {
                    if (second_matrix[x*B + y] > second_matrix[x*B + i] + first_matrix[i*B + y])
                        second_matrix[x*B + y] = second_matrix[x*B + i] + first_matrix[i*B + y];
                    __syncthreads();
                }
            }
            if (blockIdx.y != 0) {//column block
                #pragma unroll
                for (int i=0; i<B; i++) {
                    if (second_matrix[x*B + y] > first_matrix[x*B + i] + second_matrix[i*B + y])
                        second_matrix[x*B + y] = first_matrix[x*B + i] + second_matrix[i*B + y];
                    __syncthreads();
                }
            }
            
            graph_dist[global_x * vertexNum + global_y] = second_matrix[x*B + y];
        }
    } 
}

__global__ void cudaFW_phase3(int ith_round, int vertexNum, int *graph_dist, int B, int blockOffset)
{
    int blockIdx_x = blockIdx.x + blockOffset;
    int blockIdx_y = blockIdx.y;
    if (blockIdx_x != ith_round && blockIdx_y != ith_round) {
        extern __shared__ int share_space[];
        int* row_block = &share_space[0];
        int* column_block = &share_space[B*B];

        int x = threadIdx.x;
        int y = threadIdx.y;
        int global_x = blockIdx_x * blockDim.x + x;
        int global_y = blockIdx_y * blockDim.y + y;
        int k = x + ith_round * B;//correspond row block possition
        int q = y + ith_round * B;//correspond column block position

        if (global_x < vertexNum && q < vertexNum)
            row_block[x*B + y] = graph_dist[global_x * vertexNum + q];
        if (global_x >= vertexNum || q >= vertexNum)
            row_block[x*B + y] = INF;
        if (global_y < vertexNum && k < vertexNum)
            column_block[x*B + y] = graph_dist[k * vertexNum + global_y];
        if (global_y >= vertexNum || k >= vertexNum)
            column_block[x*B + y] = INF;
        __syncthreads();

        if (global_x < vertexNum && global_y < vertexNum) {
            int selfDist = graph_dist[global_x * vertexNum + global_y];
            #pragma unroll
            for (int i=0; i<B; i++) {
                if (selfDist > row_block[x*B + i] + column_block[i*B + y])
                    selfDist = row_block[x*B + i] + column_block[i*B + y];
            }
            graph_dist[global_x * vertexNum + global_y] = selfDist;
        }
    }
}

__global__ void cuda_print(int *graph_dist, int vertexNum)
{
    for (int i=0; i< vertexNum; i++) {
	for (int j=0; j< vertexNum; j++) {
	   if (graph_dist[i * vertexNum + j] >= INF) 
		printf("INF ");
	   else
		printf("%d ", graph_dist[i * vertexNum + j]);
	}
	printf("\n");
    }

}

__global__ void cuda_alignment (int *graph_dist, int *temp_dist, int blockOffset, int vertexNum)
{
    if (blockIdx.x >= blockOffset) {
        int x = blockIdx.x;
        for (int i=0; i<vertexNum; i++) {
            graph_dist[x * vertexNum + i] = temp_dist[x * vertexNum + i];
        }
    }

}

int main(int argc, char *argv[])
{
    if(argc != 4) {
        printf("not enough argument!\n");
        exit(0);
    }

    int processNum, rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processNum);

    //printf("I'm at starter\n");
    
    B = atoi(argv[3]);

    if (B > 32) 
        B = 32;

    if (rank == 0) {
        input(argv[1]);
    }
    MPI_Bcast(&vertexNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graphMap, vertexNum * vertexNum, MPI_INT, 0, MPI_COMM_WORLD);


    int round = ((vertexNum-1)+B) / B;


    int iDeviceCount = 0;
    cudaGetDeviceCount( &iDeviceCount );
    printf("I'm rank %d. my DeviceCount is %d\n", rank, iDeviceCount);
    if (iDeviceCount < 1) {
        printf("No GPU device\n");
        exit(0);
    }
    cudaSetDevice(rank);

    int* buffArray = (int*) malloc (vertexNum * vertexNum * sizeof(int));
    graphDist = (int*) malloc (vertexNum * vertexNum * sizeof(int));

    int *Dgraph_dist, *temp_dist;

    int blockOffset = ((round+1) + processNum) / processNum;

    
    dim3 blockNum_phase1(1,1);
    dim3 blockNum_phase2(round, 2);
    dim3 blockNum_phase3(blockOffset, round);
    dim3 threadNum(B, B);


    cudaMalloc((void**) &Dgraph_dist, vertexNum * vertexNum * sizeof(int));
    cudaMemcpy(Dgraph_dist, graphMap, vertexNum * vertexNum * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMalloc((void**) &temp_dist, vertexNum * vertexNum * sizeof(int));

    for (int i=0; i < round; i++) { 
        if (rank == 0) {
            cudaFW_phase1 <<< blockNum_phase1, threadNum, B*B*sizeof(int) >>>(i, vertexNum, Dgraph_dist, B);
            cudaFW_phase2 <<< blockNum_phase2, threadNum, B*B*sizeof(int)*2 >>>(i, vertexNum, Dgraph_dist, B);
           
            cudaStreamSynchronize(0);
            cudaMemcpy(buffArray, Dgraph_dist, vertexNum * vertexNum * sizeof(int), cudaMemcpyDeviceToHost);
            MPI_Send(buffArray, vertexNum * vertexNum, MPI_INT, 1, 0, MPI_COMM_WORLD);
	    
        } else if (rank == 1) {
            MPI_Recv(buffArray, vertexNum * vertexNum, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cudaMemcpy(Dgraph_dist, buffArray, vertexNum * vertexNum * sizeof(int), cudaMemcpyHostToDevice);
        }
        cudaFW_phase3 <<< blockNum_phase3, threadNum, B*B*sizeof(int)*2 >>>(i, vertexNum, Dgraph_dist, B, blockOffset * rank);
        
        cudaStreamSynchronize(0);

        if (rank == 0) {
            MPI_Recv(buffArray, vertexNum * vertexNum, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cudaMemcpy(temp_dist, buffArray, vertexNum * vertexNum * sizeof(int), cudaMemcpyHostToDevice);
            cudaStreamSynchronize(0);
            cuda_alignment <<< vertexNum, 1 >>> (Dgraph_dist, temp_dist, B * blockOffset, vertexNum);
            cudaStreamSynchronize(0);
        } else if (rank == 1) {
            cudaMemcpy(buffArray, Dgraph_dist, vertexNum * vertexNum * sizeof(int), cudaMemcpyDeviceToHost);
            MPI_Send(buffArray, vertexNum * vertexNum, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }
    if (rank == 0) {
        cudaMemcpy(graphDist, Dgraph_dist, vertexNum * vertexNum * sizeof(int), cudaMemcpyDeviceToHost);	
	
    }
    cudaFree(Dgraph_dist);
    cudaFree(temp_dist);

    if (rank == 0) {
	output(argv[2]);
    }

    MPI_Finalize();

    exit(0);
}

void input(char *inFileName)
{
    FILE *infile = fopen(inFileName, "r");
    fscanf(infile, "%d %d", &vertexNum, &edgeNum);

    for (int i=0; i<vertexNum; i++) {
        for (int j = 0; j < vertexNum; j++) {
            if (i == j)
                graphMap[i*vertexNum + j] = 0;
            else
                graphMap[i*vertexNum + j] = INF;
        }
    }

    while (--edgeNum >= 0) {
        int a, b, v;
        fscanf(infile, "%d %d %d", &a, &b, &v);
        --a, --b;
        graphMap[a * vertexNum + b] = v;
    }
}

void output(char *outFileName)
{
    FILE *outfile = fopen(outFileName, "w");
    for (int i=0; i<vertexNum; i++) {
        for (int j=0; j<vertexNum; j++) {
            if (graphDist[i*vertexNum + j] >= INF)
                fprintf(outfile, "INF ");
            else
                fprintf(outfile, "%d ", graphDist[i*vertexNum + j]);
        }
        fprintf(outfile, "\n");
    }
}

