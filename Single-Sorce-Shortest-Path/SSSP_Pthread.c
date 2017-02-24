//SSSP_Pthread

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int ThreadNum, vertexNum, start_point;
int** mapArray;
int* queue;
const int MAXNUM = 2147483647;
pthread_mutex_t mutex;
int queueIndex=0;

typedef struct vertexDict
{
    int father;
    int dist;
    int tag;
} Vdict;
Vdict* vertexDict;

typedef struct Argument
{
    int begin_id;
    int node_id;
} Arg;

void* parallel_init(void* id)
{
    int k=*(int*) id;
    int i,j;
    for (i=(k+1); i<(vertexNum+1); i+=ThreadNum) {
        if (i!=start_point) {
            pthread_mutex_lock(&mutex);
            queue[queueIndex] = i;
            queueIndex++;
            pthread_mutex_unlock(&mutex);
        }
        vertexDict[i].dist = MAXNUM;
        vertexDict[i].father = -1;
        vertexDict[i].tag = 0;
        for (j=1; j<(vertexNum+1); j++) {
            mapArray[i][j] = MAXNUM;
            if (j==i)
                mapArray[i][j] = 0;
        }
    }
    pthread_exit(NULL);
}

void init()
{
    vertexDict = malloc ((vertexNum+1)*sizeof(Vdict));
    queue = malloc ((vertexNum-1)*sizeof(int));

    mapArray = malloc ((vertexNum+1)*sizeof(int*));
    int column;
    for (column=1; column<(vertexNum+1); column++) {
        mapArray[column] = malloc ((vertexNum+1)*sizeof(int));
    }

    pthread_t threads[ThreadNum];
    int* id = malloc (ThreadNum*sizeof(int));
    int i;

    for (i=0; i<ThreadNum; i++) {
        id[i] = i;
        pthread_create(&threads[i], NULL, parallel_init, (void*) &id[i]);
    }

    for (i=0; i<ThreadNum; i++)
        pthread_join(threads[i], NULL);
}

void* relax(void* arg)
{
    int thread_ID = ((Arg*) arg)->begin_id;
    int node_ID = ((Arg*) arg)->node_id;
    int i;
    for (i=(thread_ID+1); i<(vertexNum+1); i+=ThreadNum) {
        if (vertexDict[i].tag == 1)
            continue;
        if (mapArray[node_ID][i] == MAXNUM || mapArray[node_ID][i] + vertexDict[node_ID].dist < 0)
            continue;
        if (mapArray[node_ID][i]+vertexDict[node_ID].dist < vertexDict[i].dist) {
            vertexDict[i].dist = mapArray[node_ID][i]+vertexDict[node_ID].dist;
            vertexDict[i].father = node_ID;
        }
    }

    pthread_exit(NULL);
}

void shortest_Path()
{
    int i;
    for (i=1; i<(vertexNum+1); i++) {
        if (i!=start_point && mapArray[start_point][i]<MAXNUM) {
            vertexDict[i].father = start_point;
            vertexDict[i].dist = mapArray[start_point][i];
        }
    }

    
    vertexDict[start_point].father = start_point;
    vertexDict[start_point].dist = 0;
    vertexDict[start_point].tag = 1;

    while (queueIndex>0) {
        int min = MAXNUM;
        int vertex_x;
        int i;
        for (i=0; i<(vertexNum-1); i++) {
            if (vertexDict[queue[i]].tag==0 && vertexDict[queue[i]].dist < min) {
                min = vertexDict[queue[i]].dist;
                vertex_x = queue[i];
            }
        }


        vertexDict[vertex_x].tag = 1;
        queueIndex--;

        Arg* arg = malloc(ThreadNum*sizeof(Arg));

        pthread_t threads[ThreadNum];
        for (i=0; i<ThreadNum; i++) {
            arg[i].begin_id = i;
            arg[i].node_id = vertex_x;
            pthread_create(&threads[i], NULL, relax, &arg[i]);
        }

        for (i=0; i<ThreadNum; i++)
            pthread_join(threads[i], NULL);

        free(arg); 
    }

}

int main(int argc, char* argv[])
{
    int edgeNum;
    FILE *fin, *fout;

    if (argc != 5) {
        printf("not enough argument!\n");
        exit(0);
    }

    ThreadNum = atoi (argv[1]);
    start_point = atoi (argv[4]);

    fin = fopen (argv[2], "r");
    fscanf (fin, "%d %d", &vertexNum, &edgeNum);

    pthread_mutex_init(&mutex, NULL);

    init();   
    
    //finv1: fin vertex 1, finv2: fin vertex 2, finw: fin edge weight
    int finv1, finv2, finw;
    int i;
    for (i=0; i<edgeNum; i++) {
        fscanf(fin, "%d %d %d\n", &finv1, &finv2, &finw);
        mapArray[finv1][finv2] = finw;
        mapArray[finv2][finv1] = finw;
    }  

    shortest_Path();

    fout = fopen(argv[3], "w");
    int j;
    for (i=1; i<(vertexNum+1); i++) {
        int* printArray = malloc (vertexNum*sizeof(int));
        int k=0;
        int f;
        printArray[k] = i;
        f = vertexDict[i].father;
        while (f!=start_point) {
            k++;
            printArray[k] = f;
            f = vertexDict[f].father;
        }

        fprintf(fout, "%d ", start_point);

        for (j=k; j>=0; j--) {
            fprintf(fout, "%d ", printArray[j]);
        }
        fprintf(fout, "\n");
        free(printArray);
    }

    fclose(fin);
    fclose(fout);
    
    exit(0);

}
