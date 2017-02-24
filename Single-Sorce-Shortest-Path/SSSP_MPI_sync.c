#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int MAXNUM = 2147483647;
int queueIndex=0;
int vertexID;
int vertexNum;
int startPoint;
int* selfInfo;
int** mapArray;

int* nullMessage;

void malloc2dint(int*** array, int n, int m) 
{
    int* p = malloc (n*m*sizeof(int));
    (*array) = malloc (n*sizeof(int*));
    int i;
    for (i=0; i<n; i++)
	(*array)[i] = &(p[i*m]);

}

void mpi_shortest_path() 
{
    int neighborCount = 0;
    int tagSum = 0;
    int i,j;
    for (i=1; i<(vertexNum+1); i++) 
        if (vertexID!=i && mapArray[vertexID][i] < MAXNUM) 
            neighborCount++;
    int* neighborList = malloc (neighborCount*sizeof(int));
    int* sendRequestList = malloc (neighborCount*sizeof(int));
    int* recvRequestList = malloc (neighborCount*sizeof(int));
    int** neighborDict;
    malloc2dint(&neighborDict, neighborCount, 4);
    j=0;
    for (i=1; i<(vertexNum+1); i++) {
        if (vertexID!=i && mapArray[vertexID][i] < MAXNUM) {
            sendRequestList[j] = j;
            recvRequestList[j] = j;
            neighborList[j] = i;
            neighborDict[j][0] = i;
            neighborDict[j][1] = -1;
            neighborDict[j][2] = MAXNUM;
            neighborDict[j][3] = 0;
            j++;
        }
    }
    
    selfInfo = malloc (4*sizeof(int));
    if (vertexID!=startPoint) {
        selfInfo[0] = vertexID;
        selfInfo[1] = -1;
        selfInfo[2] = MAXNUM;
        selfInfo[3] = 0;
    } else {
        selfInfo[0] = vertexID;
        selfInfo[1] = vertexID;
        selfInfo[2] = 0;
        selfInfo[3] = 0;
    } 	

    while (tagSum<vertexNum) {
        tagSum = 0;
			
        for (i=0; i<neighborCount; i++) {
            if (selfInfo[3] == 1)
                MPI_Isend(&(nullMessage[0]), 4, MPI_INT, (neighborList[i]-1), (vertexID-1), MPI_COMM_WORLD, &sendRequestList[i]);
            else if (selfInfo[1] == neighborList[i])
                MPI_Isend(&(nullMessage[0]), 4, MPI_INT, (neighborList[i]-1), (vertexID-1), MPI_COMM_WORLD, &sendRequestList[i]);
            else
                MPI_Isend(&(selfInfo[0]), 4, MPI_INT, (neighborList[i]-1), (vertexID-1), MPI_COMM_WORLD, &sendRequestList[i]);
        }
	
        for (i=0; i<neighborCount; i++) {
            MPI_Irecv(&(neighborDict[i][0]), 4, MPI_INT, (neighborList[i]-1), (neighborList[i]-1), MPI_COMM_WORLD, &recvRequestList[i]);
        }
        for (i=0; i<neighborCount; i++) {
            MPI_Wait(&sendRequestList[i], MPI_STATUS_IGNORE);
        }

        for (i=0; i<neighborCount; i++) {
            MPI_Wait(&recvRequestList[i], MPI_STATUS_IGNORE);
        }

        selfInfo[3] = 1;
        
        for (i=0; i<neighborCount; i++) {
            if (neighborDict[i][3] == -1)
                continue;
            else if (neighborDict[i][2] == MAXNUM || neighborDict[i][2]+mapArray[vertexID][neighborDict[i][0]] < 0)
                continue;
            else if (neighborDict[i][2]+mapArray[vertexID][neighborDict[i][0]] < selfInfo[2]) {
                selfInfo[2] = neighborDict[i][2]+mapArray[vertexID][neighborDict[i][0]];
                selfInfo[1] = neighborDict[i][0];
                selfInfo[3] = 0;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&selfInfo[3], &tagSum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[])
{
    FILE *inputFile, *outputFile;
    int rank, processNum;
    int edgeNum;
    int column, i, j;
 
    if (argc!=5) {
        printf("Not enough argument!\n");
        exit(0);
    }

    int rc = MPI_Init(&argc, &argv);

    if (rc!=MPI_SUCCESS) {
        printf("Error starting MPI program. Abort!\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processNum);

    if (rank == 0) {
        inputFile = fopen (argv[2], "r");
        fscanf (inputFile, "%d %d \n", &vertexNum, &edgeNum);
        printf("%d %d\n", vertexNum, edgeNum);	
    }

    MPI_Bcast(&vertexNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (processNum != vertexNum) {
        printf("Not same number of vertex and process!\n");
        MPI_Finalize();
        exit(0);
    }

    malloc2dint(&mapArray, (vertexNum+1), (vertexNum+1));
    for (i=1; i<(vertexNum+1); i++) {
        for (j=1; j<(vertexNum+1); j++) {
            mapArray[i][j] = MAXNUM;
            if (j==i)
                mapArray[i][j] = 0;
        }
    }

    vertexID = rank+1;
    

    startPoint = atoi (argv[4]);
    nullMessage = malloc (4*sizeof(int));
    nullMessage[0] = 0;//selfID
    nullMessage[1] = -1;//father
    nullMessage[2] = -1;//dist
    nullMessage[3] = -1;//tag

    if (rank==0) {
        int finv1, finv2, finw;
        for (i=0; i<edgeNum; i++) {
            fscanf(inputFile, "%d %d %d\n", &finv1, &finv2, &finw);
            mapArray[finv1][finv2] = finw;
            mapArray[finv2][finv1] = finw;
        }
	 
    } 
    MPI_Bcast(&(mapArray[0][0]), (vertexNum+1)*(vertexNum+1), MPI_INT, 0, MPI_COMM_WORLD);	
    MPI_Barrier(MPI_COMM_WORLD);
    
    mpi_shortest_path();


    if (rank != 0) {
        MPI_Send(&(selfInfo[0]), 4, MPI_INT, 0, rank, MPI_COMM_WORLD);
    } else {
        int** vertexDict;
	    malloc2dint(&vertexDict, (vertexNum+1), 4);
        
        for (i=0; i<4; i++) 
            vertexDict[1][i] = selfInfo[i];

        for (i=1; i<processNum; i++) 
            MPI_Recv(&(vertexDict[(i+1)][0]), 4, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	for (i=1; i<vertexNum+1; i++) {
		for (j=0; j<4; j++)
		    printf("%d ", vertexDict[i][j]);
	    printf("\n");
	}
        outputFile = fopen(argv[3], "w");	

        for (i=1; i<(vertexNum+1); i++) {
            int* printArray = malloc (vertexNum*sizeof(int));
            int k=0;
            int f;
            printArray[k] = i;
            f = vertexDict[i][1];
            while (f!=startPoint) {
                k++;
                printArray[k] = f;
                f = vertexDict[f][1];
            }

            fprintf(outputFile, "%d ", startPoint);

            for (j=k; j>=0; j--) {
                fprintf(outputFile, "%d ", printArray[j]);
            }

            fprintf(outputFile, "\n");
            free(printArray);
        }

        fclose(inputFile);
        fclose(outputFile);

    }

    MPI_Finalize();
    exit(0);
    
}
