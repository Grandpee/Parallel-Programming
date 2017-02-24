#include "mpi.h"
#include <iostream>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

using namespace std;


double TotalTimeStart, TotalTimeEnd;
double IOReadTimeStart, IOReadTimeEnd;
double IOWriteTimeStart, IOWriteTimeEnd;
double CommTimeStart, CommTimeEnd;
double CommTime=0;
double IOTime=0;
double TotalTime=0;

void eoSort(int inputArray[], int smallArraySize) 
{
	bool ended = false;

	while(!ended) {
		ended = true;

		for (int j=0; j< 2; j++) {
			if (j == 0) {				//even state
				for (int i=0; i < smallArraySize; i+=2)
				{
					if (inputArray[i] > inputArray[i+1] && i+1 < smallArraySize) {
						swap(inputArray[i], inputArray[i+1]);
						ended = false;
					}
				}
			} else {							//odd state
				for (int i=1; i < smallArraySize; i+=2) {
					if (inputArray[i] > inputArray[i+1] && i+1 < smallArraySize) {
						swap(inputArray[i], inputArray[i+1]);
						ended = false;
					}
				}
			}
		} // end one time of even odd exchange
	}
}

//for rank that bigger
int rank_bigger( int rankNum, int smallNum)
{
	int pal = rankNum - 1;
	int buffer1;
	int buffer2 = smallNum;
	
	CommTimeStart = MPI_Wtime();
	MPI_Recv(&buffer1, 1, MPI_INT, pal, pal, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Send(&buffer2, 1, MPI_INT, pal, rankNum, MPI_COMM_WORLD);
	CommTimeEnd = MPI_Wtime();
	
	CommTime += (CommTimeEnd - CommTimeStart);
	
	if(buffer1 > smallNum) 
		swap(buffer1, smallNum);

	return smallNum;
}

//for rank that smaller
int rank_smaller( int rankNum, int largeNum)
{
	int pal = rankNum + 1;
	int buffer1 = largeNum;
	int buffer2;
	
	MPI_Send(&buffer1, 1, MPI_INT, pal, rankNum, MPI_COMM_WORLD);
	MPI_Recv(&buffer2, 1, MPI_INT, pal ,pal, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	if (buffer2 < largeNum)
		swap(buffer2, largeNum);

	return largeNum;
}

int findExtremeNum (int findArray[], int size, int largeSmallFlag)
{	
	int temp;
	int n=0;
	if (largeSmallFlag == 0) {			//find smallestNum
		temp = findArray[0];
		for (int i=1; i<size; i++) {
			if (findArray[i] < temp){
				temp = findArray[i];
				n=i;
				n=i;
			}
		}
	} else {							//find largestNum
		temp = findArray[0];
		for (int i=1; i<size; i++) {
			if (findArray[i] > temp) {
				temp = findArray[i];
				n=i;
			}
		}
	}
	return n;
}

void mpi_eoSort(MPI_File *input, MPI_File *output, const int lrank, const int size, const int N)
{
	MPI_Offset startpoint;
	int shiftArray[1];
	int scale = N/size;
	startpoint = lrank * scale * sizeof(shiftArray[0]);
	if (lrank == size-1)
		scale = (N-1) - scale*lrank + 1; // (N-1) is endpoint
	int lArray[scale];
	
	IOReadTimeStart = MPI_Wtime();
	MPI_File_read_at(*input, startpoint, &lArray, scale, MPI_INT, MPI_STATUS_IGNORE);
	IOReadTimeEnd = MPI_Wtime();	

	int reduceSum = 0;
	int reduceNum = 0;

	while(reduceSum != (size+1)) {
		reduceSum = 0;
		reduceNum = 0;

		//even odd two state
		for (int mpi_eoState = 0; mpi_eoState < 2; mpi_eoState++) {
			int tempNum = 0;
			int k=0;
			if (mpi_eoState == 0) {									//even State
				if (lrank%2 == 0 && (lrank+1) < size) {
					k = findExtremeNum(lArray, scale, 1);
					tempNum = lArray[k];
					lArray[k] = rank_smaller(lrank,lArray[k]);
					if (tempNum == lArray[k])
						reduceNum += 1;
				} else if (lrank%2 == 0 && (lrank+1) == size) {		//case of the last rank that is even
					reduceNum += 1;
				} else if (lrank%2 == 1) {
					k = findExtremeNum(lArray, scale, 0);
					lArray[k] = rank_bigger(lrank, lArray[k]);
				}
			} else {												//odd State
				if (lrank%2 == 1 && (lrank+1) < size) {
					k = findExtremeNum(lArray, scale, 1);
					tempNum = lArray[k];
					lArray[k] = rank_smaller(lrank, lArray[k]);
					if (tempNum == lArray[k])
						reduceNum += 1;
				} else if ( lrank == 0 || (lrank%2==1 && (lrank+1) == size)) {    //boundary condition
					reduceNum += 1;
				} else if (lrank%2 == 0 && lrank != 0) {
					k = findExtremeNum(lArray, scale, 0);
					lArray[k] = rank_bigger(lrank, lArray[k]);
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);

		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&reduceNum, &reduceSum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	eoSort(lArray, scale);
	
	IOWriteTimeStart = MPI_Wtime();
	MPI_File_write_at(*output, startpoint, &lArray, scale, MPI_INT, MPI_STATUS_IGNORE);
	IOWriteTimeEnd = MPI_Wtime();
}

int main(int argc, char *argv[])
{
	MPI_File inputFile, outputFile;
	int rank, size;
	int fileOpenError;
	int N;	//input array size

	
	int rc = MPI_Init(&argc, &argv);
	
	if(rc != MPI_SUCCESS) {
		cout << "Error starting MPI program. Terminateing." << endl;
		MPI_Abort(MPI_COMM_WORLD, rc);
		return 0;
	}
	
	TotalTimeStart = MPI_Wtime();
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	//argument check
	if(argc != 4) {
		cout << "Not enough argument!" << endl;
		MPI_Finalize();
		return 0;
	}

	//open input file
	fileOpenError = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &inputFile);
	if (fileOpenError) {
		cout << "open input file error" << endl;
		MPI_Finalize();
		return 0;
	}

	//open output file
	fileOpenError = MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &outputFile);
	if (fileOpenError) {
		cout << "open output file error" << endl;
		MPI_Finalize();
		return 0;
	}

	N = atoi (argv[1]);

	if ( N > size) {
		mpi_eoSort(&inputFile, &outputFile, rank, size, N);
	} else {
		if (rank == 0) {
			int Array[N];
			IOReadTimeStart = MPI_Wtime();
			MPI_File_read_at(inputFile, 0, &Array, N, MPI_INT, MPI_STATUS_IGNORE);
			IOReadTimeEnd = MPI_Wtime();
			eoSort(Array, N);
			IOWriteTimeStart = MPI_Wtime();
			MPI_File_write_at(outputFile, 0, &Array, N, MPI_INT, MPI_STATUS_IGNORE);
			IOWriteTimeEnd = MPI_Wtime();
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	

	MPI_File_close(&inputFile);
	MPI_File_close(&outputFile);
	
	TotalTimeEnd = MPI_Wtime();
	
	
	//calculate Time
	IOTime = IOReadTimeEnd - IOReadTimeStart + IOWriteTimeEnd - IOWriteTimeStart;
	TotalTime = TotalTimeEnd - TotalTimeStart;
	
	double overallIOTime=0;
	double overallTotalTime=0;
	double overallCommTime=0;
	
	MPI_Reduce(&IOTime, &overallIOTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&TotalTime, &overallTotalTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&CommTime, &overallCommTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if (rank == 0) {
		double avgTotalTime = overallTotalTime/size;
		double avgIOTime = overallIOTime/size;
		double avgCommTime = 2*overallCommTime/size;
		
		cout << "The basic version Totaltime is " << avgTotalTime << " IOTime is " << avgIOTime << " CommTime is " << avgCommTime << endl;
	}
	
	MPI_Finalize();
	return 0;
}

