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

void mpi_shortest_path(int rank) 
{
    int neighborCount = 0;
    int i,j;
    int token=0, tokenSlot=0;//0 for not done, 1 for done
    int flag = 0;//0 for not send to smaller rank, 1 for otherwise
    int terminatedSignal = 0;
    int noSend = 0;//for largest vertex
    int genToken = 0;//1 for generated, 0 otherwise
    int tokenSendReq = 20, tokenRecvReq = 19;
    for (i=1; i<(vertexNum+1); i++) 
        if (vertexID!=i && mapArray[vertexID][i] < MAXNUM) 
            neighborCount++;
    int* neighborList = malloc (neighborCount*sizeof(int));
    int* sendRequestList = malloc (neighborCount*sizeof(int));
    int* recvRequestList = malloc (neighborCount*sizeof(int));
    int** neighborDict;
    malloc2dint(&neighborDict, neighborCount, 5);
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
            neighborDict[j][4] = 0;//terminatedsignal
            j++;
        }
    }
    
    selfInfo = malloc (5*sizeof(int));
    if (vertexID!=startPoint) {
        selfInfo[0] = vertexID;
        selfInfo[1] = -1;
        selfInfo[2] = MAXNUM;
        selfInfo[3] = 0;
        selfInfo[4] = 0;//terminatedsignal
    } else {
        selfInfo[0] = vertexID;
        selfInfo[1] = vertexID;
        selfInfo[2] = 0;
        selfInfo[3] = 0;
        selfInfo[4] = 0;//terminatedsignal
    } 	
    int k = 0;
    while (1) {
		
        for (i=0; i<neighborCount; i++) {
            if (neighborDict[i][4] == 0) {
                if (selfInfo[3] == 1)
                    MPI_Isend(&(nullMessage[0]), 5, MPI_INT, (neighborList[i]-1), (vertexID-1), MPI_COMM_WORLD, &sendRequestList[i]);
                else if (selfInfo[1] == neighborList[i])
                    MPI_Isend(&(nullMessage[0]), 5, MPI_INT, (neighborList[i]-1), (vertexID-1), MPI_COMM_WORLD, &sendRequestList[i]);
                else {
                    MPI_Isend(&(selfInfo[0]), 5, MPI_INT, (neighborList[i]-1), (vertexID-1), MPI_COMM_WORLD, &sendRequestList[i]);
                    if (vertexID > neighborList[i])
                        flag = 1;
                }
            }
        }
	
        for (i=0; i<neighborCount; i++) {
            if (neighborDict[i][4] == 0)
                MPI_Irecv(&(neighborDict[i][0]), 5, MPI_INT, (neighborList[i]-1), (neighborList[i]-1), MPI_COMM_WORLD, &recvRequestList[i]);
        }
	
        for (i=0; i<neighborCount; i++) {
            if (neighborDict[i][4] == 0)
                MPI_Wait(&sendRequestList[i], MPI_STATUS_IGNORE);
        }
	
        for (i=0; i<neighborCount; i++) {
            if (neighborDict[i][4] == 0)
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
        
        //rank > 0 will do
        if (flag == 1 && tokenSlot == 1) {
            tokenSlot = -1;
            flag = 0;
        }
        
        //only rank 0 will do 
        if (rank == 0) {
            if (selfInfo[3] == 1 && tokenSlot == 0 && genToken == 0) {
                tokenSlot = 1;
                genToken = 1;
            } else if (selfInfo[3] == 1 && tokenSlot == 1) {
                tokenSlot = 2;
		k = 1;
            } else if (selfInfo[3] == 1 && tokenSlot == -1) {
                tokenSlot = 1;
            } else if (selfInfo[3] == 0 && tokenSlot != 0) {
                tokenSlot = 0;
                genToken = 0;
            }
        }

        if (terminatedSignal == 1)
            break;

        if (tokenSlot == 2) {
            selfInfo[4] = 1;
            nullMessage[4] = 1;
        }
	
        //printf("I'm vertex %d, tag %d, token: %d, T: %d, noSend: %d, genToken: %d, flag: %d\n", vertexID, selfInfo[3], tokenSlot, terminatedSignal, noSend, genToken, flag);
        if (selfInfo[3] == 1 && tokenSlot != 0) {
	    
            if (rank == vertexNum-1 && noSend == 0) {
                if (tokenSlot == 1)
                    noSend = 1;
                MPI_Isend(&tokenSlot, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &tokenSendReq);

            }
            else if (rank != vertexNum-1)
                MPI_Isend(&tokenSlot, 1, MPI_INT, rank+1, rank, MPI_COMM_WORLD, &tokenSendReq);
            tokenSlot = 0;
	    
        } else {
	    
            if (rank == vertexNum-1 && noSend == 0) 
                MPI_Isend(&token, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &tokenSendReq);
            else if (rank != vertexNum-1)
                MPI_Isend(&token, 1, MPI_INT, rank+1, rank, MPI_COMM_WORLD, &tokenSendReq);
	    

        }
	

        if (selfInfo[4] == 0) {
            if (rank == 0)
                MPI_Irecv(&token, 1, MPI_INT, vertexNum-1, vertexNum-1, MPI_COMM_WORLD, &tokenRecvReq);
            else
                MPI_Irecv(&token, 1, MPI_INT, rank-1, rank-1, MPI_COMM_WORLD, &tokenRecvReq);
        }
        if (noSend == 0)
            MPI_Wait(&tokenSendReq, MPI_STATUS_IGNORE);
	
	
        if (selfInfo[4] == 0)
            MPI_Wait(&tokenRecvReq, MPI_STATUS_IGNORE);
	//printf("I'm vertex %d. I'm after token send\n", vertexID);
	
        if (token != 0) {
            tokenSlot = token;
            token = 0;
        }
    }
    printf("I'm vertex %d, I'm out\n", vertexID);
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
    nullMessage = malloc (5*sizeof(int));
    nullMessage[0] = 0;//selfID
    nullMessage[1] = -1;//father
    nullMessage[2] = -1;//dist
    nullMessage[3] = -1;//tag
    nullMessage[4] = 0;//terminatedsignal

    if (rank==0) {
        int finv1, finv2, finw;
        for (i=0; i<edgeNum; i++) {
            fscanf(inputFile, "%d %d %d\n", &finv1, &finv2, &finw);
            mapArray[finv1][finv2] = finw;
            mapArray[finv2][finv1] = finw;
        }
	 
    } 
    MPI_Bcast(&(mapArray[0][0]), (vertexNum+1)*(vertexNum+1), MPI_INT, 0, MPI_COMM_WORLD);	
    
    mpi_shortest_path(rank);
    //printf("I'm vertex %d, My selfInfo is: \n", vertexID);
    //    for (i=0; i<4; i++) {
    //        printf("%d ", selfInfo[i]);
    //    }
    //printf("\n");

    //printf("I'm vertex %d and rank is %d, I'm about to send or recv self info to rank 0!\n", vertexID, rank);
    if (rank != 0) {
	MPI_Send(&(selfInfo[0]), 5, MPI_INT, 0, 87, MPI_COMM_WORLD);
	
    } else {
        int** vertexDict;
	    malloc2dint(&vertexDict, (vertexNum+1), 5);
	
	
        for (i=0; i<5; i++) 
            vertexDict[1][i] = selfInfo[i];

        for (i=1; i<processNum; i++) 
            MPI_Recv(&(vertexDict[(i+1)][0]), 5, MPI_INT, i, 87, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	//for (i=1; i<vertexNum+1; i++) {
	//    for (j=0; j<4; j++)
	//	printf("%d ", vertexDict[i][j]);
	//    printf("\n");
	//}
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
    //printf("I'm vertex %d and rank is %d, I'm done\n", vertexID, rank);

    MPI_Finalize();
    exit(0);
    
}
