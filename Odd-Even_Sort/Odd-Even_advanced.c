#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int compare(const void *s1, const void *s2);
void rank_bigger_adv(int* bigger_rank_Array, int rankNum, int size);
int rank_smaller_adv(int* smaller_rank_Array, int rankNum, int size);
void mpi_advSort(MPI_File *input, MPI_File *output, const int lrank, const int size, const int N);

int main(int argc, char *argv[])
{
    MPI_File intputFile, outputFile;
    int rank, size;
    int fileOpenError;
    int N;

    int rc = MPI_Init(&argc, &argv);

    if (rc!=MPI_SUCCESS) {
        printf("Error starting MPI program. Abort!\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc!=4) {
        printf("Not enough argument!\n");
        MPI_Finallize();
        exit(0);
    }

    fileOpenError = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &inputFile);
    if (fileOpenError) {
        printf("Open input file error.\n");
        MPI_Finallize();
        exit(0);
    }

    fileOpenError = MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &outputFile);
    if (fileOpenError) {
        printf("Open output file error.\n");
        MPI_Finallize();
        exit(0);
    }

    N = atoi (argv[1]);

    if (N>size) {
        mpi_advSort(&inputFile, &outputFile, rank, size, N);
    } else {
        if (rank == 0) {
            int* Array = malloc (N*sizeof(int));
            MPI_File_read_at(intputFile, 0, &Array, N, MPI_INT, MPI_STATUS_IGNORE);
            qsort(Array, N, sizeof(int), compare);
            MPI_File_write_at(outputFile, 0, &Array, N, MPI_INT, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_File_close(&inputFile);
    MPI_File_close(&outputFile);

    MPI_Finallize();
    exit(0);
}

int compare(const void *s1, const void *s2)
{
    if (*(int*)s1 > *(int*)s2)
        return 1;
    else if (*(int*s1) < *(int*)s2)
        return (-1);
    else
        return 0;
}

void rank_bigger_adv(int* bigger_rank_Array, int rankNum, int size)
{
    int pal=rankNum-1;
    int palSize;

    MPI_Recv(&palSize, 1, MPI_INT, pal, pal, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&size, 1, MPI_INT, pal, rankNum, MPI_COMM_WORLD);

    int* palArray = malloc (palSize*sizeof(int));

    MPI_Sendrecv(bigger_rank_Array, size, MPI_INT, pal, rankNum, palArray, palSize, MPI_INT, pal, pal, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int mergeSize = palSize+size;
    int* mergeArray = malloc (mergeSize*sizeof(int));
    int p=0, q=0;
    int k=0;

    while (p<palSize && q<size) {
        if (palArray[p] <= bigger_rank_Array[q]) {
            mergeArray[k] = palArray[p];
            p++;
        } else {
            mergeArray[k] = bigger_rank_Array[q];
            q++;
        }
        k++;
    }

    while (q<size) {
        mergeArray[k] = bigger_rank_Array[q];
        k++; q++;
    }
    
    while (p<palSize) {
        mergeArray[k] = palArray[p];
        k++; p++;
    }

    for (int i=0; i<size; i++) {
        bigger_rank_Array[i] = mergeArray[palSize+i];
    }
}

int rank_smaller_adv(int* smaller_rank_Array, int rankNum, int size)
{
    int pal = rankNum+1;
    int palSize;
    int exchangeFlag = 1;
    MPI_Send(&size, 1, MPI_INT, rankNum, MPI_COMM_WORLD);
    MPI_Recv(&palSize, 1, MPI_INT, pal, pal, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int* palArray = malloc (palSize*sizeof(int));

    MPI_Sendrecv(smaller_rank_Array, size, MPI_INT, pal, rankNum, palArray, palSize, MPI_INT, pal, pal, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int mergeSize = size+palSize;
    int* mergeArray = malloc (mergeSize*sizeof(int));
    int p=0, q=0;
    int k=0;

    while (p<size && q<palSize) {
        if (smaller_rank_Array[p] <= palArray[q]) {
            mergeArray[k] = smaller_rank_Array[p];
            p++;
        } else {
            mergeArray[k] = palArray[q];
            if (exchangeFlag == 1 && k<size)
                exchangeFlag = 0;
            q++;
        }
        k++;
    }

    while(p<size) {
        mergeArray[k] = smaller_rank_Array[p];
        k++; p++;
    }

    while(q<palSize) {
        mergeArray[k] = palArray[q];
        k++; q++;
    }

    for (int i=0; i<size; i++) {
        smaller_rank_Array[i] = mergeArray[i];
    }

    return exchangeFlag;
}

void mpi_advSort(MPI_File *input, MPI_File *output, const int lrank, const int size, const int N)
{
    MPI_Offset = startpoint;
    int scale = N/size;
    startpoint = lrank*scale*sizeof(int);
    if (lrank==size-1) 
        scale = (N-1) - scale*lrank + 1;
    int* lArray = malloc (scale*sizeof(int));

    MPI_File_read_at(*input, startpoint, &lArray, scale, MPI_INT, MPI_STATUS_IGNORE);

    qsort(lArray, scale, sizeof(int), compare);

    MPI_Barrier(MPI_COMM_WORLD);

    int reduceNum = 0;
    int reduceSum = 0;

    while (reduceSum != (size+1)) {
        reduceNum = 0;
        reduceSum = 0;

        for (int mpi_eoState = 0; mpi_eoState < 2; mpi_eoState++) {
            if (mpi_eoState == 0) {
                if (lrank%2 == 0 && (lrank+1) < size) {
                    reduceNum += rank_smaller_adv(lArray, lrank, scale);
                } else if (lrank%2 == 0 && (lrank+1) == size) {
                    reduceNum += 1;
                } else if (lrank%2 == 1) {
                    rank_bigger_adv(lArray, lrank, scale);
                }
            } else {
                if (lrank%2 == 1 && (lrank+1) < size) {
                    reduceNum += rank_smaller_adv(lArray, lrank, scale);
                } else if (lrank == 0 || (lrank%2==1 && (lrank+1) == size)) {
                    reduceNum += 1;
                } else if (lrank%2 == 0 && lrank !=0) {
                    rank_bigger_adv(lArray, lrank, scale);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&reduceNum, &reduceSum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_File_write_at(*output, startpoint, &lArray, scale, MPI_INT, MPI_STATUS_IGNORE);

}
