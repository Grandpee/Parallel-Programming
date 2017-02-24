#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

__global__ void cudaFW_phase3(int ith_round, int vertexNum, int *graph_dist, int B)
{
    if (blockIdx.x != ith_round && blockIdx.y != ith_round) {
        extern __shared__ int share_space[];
        int* row_block = &share_space[0];
        int* column_block = &share_space[B*B];

        int x = threadIdx.x;
        int y = threadIdx.y;
        int global_x = blockIdx.x * blockDim.x + x;
        int global_y = blockIdx.y * blockDim.y + y;
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

int main(int argc, char *argv[])
{
    if(argc != 4) {
        printf("not enough argument!\n");
        exit(0);
    }

    input(argv[1]);
    B = atoi(argv[3]);

    if (B > 32)
        B = 32;

    int *Dgraph_dist;
    int round = ((vertexNum-1)+B) / B;
    graphDist = (int*) malloc (vertexNum * vertexNum * sizeof(int));


    int iDeviceCount = 0;
    cudaGetDeviceCount( &iDeviceCount );
    if (iDeviceCount < 1) {
        printf("No GPU device\n");
        exit(0);
    }
    cudaSetDevice(0);
     
    cudaMalloc((void**) &Dgraph_dist, vertexNum * vertexNum * sizeof(int));
    cudaMemcpy(Dgraph_dist, graphMap, vertexNum * vertexNum * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockNum_phase1(1,1);
    dim3 blockNum_phase2(round, 2);
    dim3 blockNum_phase3(round, round);
    dim3 threadNum(B, B);

    for (int i=0; i < round; i++) {
        cudaFW_phase1 <<< blockNum_phase1, threadNum, B*B*sizeof(int) >>>(i, vertexNum, Dgraph_dist, B);
        cudaFW_phase2 <<< blockNum_phase2, threadNum, B*B*sizeof(int)*2 >>>(i, vertexNum, Dgraph_dist, B);
        cudaFW_phase3 <<< blockNum_phase3, threadNum, B*B*sizeof(int)*2 >>>(i, vertexNum, Dgraph_dist, B);
    }

    cudaMemcpy(graphDist, Dgraph_dist, vertexNum * vertexNum * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(Dgraph_dist);
    
    output(argv[2]);

    free(graphDist);

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

