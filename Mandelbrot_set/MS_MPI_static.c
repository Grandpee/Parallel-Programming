//MPI_static

#include <X11/Xlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

typedef struct complextype
{
    double real, imag;
} Compl;

int draw_color(Compl c)
{
    int count, max;
    Compl z;
    double temp, lengthsq;

    max = 100000;
    z.real = 0.0;
    z.imag = 0.0;
    count = 0;
    lengthsq = 0.0;

    while ((lengthsq < 4.0) && (count < max)) {
        temp = z.real*z.real - z.imag*z.imag + c.real;
        z.imag = 2*z.real*z.imag + c.imag;
        z.real = temp;
        lengthsq = z.real*z.real + z.imag*z.imag;
        count++;
    }

    return count;
}

void slave(int lrank, int height, double scale_real, double scale_imag, double real_min, double imag_min) 
{
    int selfWidth;
    int selfStartPoint;

    MPI_Recv(&selfWidth, 1, MPI_INT, 0, lrank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&selfStartPoint, 1, MPI_INT, 0, lrank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int* columnArray = malloc ((height+1)*sizeof(int));
    Compl c;   

    for (int a=selfStartPoint; a<selfStartPoint+selfWidth; a++) {
        for (int b=0; b<height; b++) {
           c.real = real_min + (double) a * scale_real;
           c.imag = imag_min + (double) b * scale_imag;
           columnArray[b] = draw_color(c);
        }
        columnArray[height] = a;
        MPI_Send(columnArray, height+1, MPI_INT, 0, 88, MPI_COMM_WORLD);
    }
}

void master(double left_R_range, double low_I_range, double scale_real, double scale_imag, int width, int height, int num_process, int enable_X)
{
    int* totalArray = malloc ((width*height)*sizeof(int));

    double timeStart, timeEnd;
    
    timeStart = MPI_Wtime();
    if (width < num_process-1 || num_process == 1) {
        int i, j;
        Compl c;
        int repeats;
        for (i=0; i<width; i++) {
            for (j=0; j<height; j++) {
                c.real = left_R_range + ((double) i * scale_real);
                c.imag = low_I_range + ((double) j * scale_imag);
                repeats = draw_color(c);
		        
                totalArray[i*height+j] = repeats;
            }
        }
    }
    else {
	//printf("int width larger function\n");
        int localWidth;
        int startpoint;
        int* receivedArray = malloc ((height+1)*sizeof(int));
	int k;
        for (k=1; k<num_process; k++) {
            if (k!=(num_process-1)) {
                localWidth = width/(num_process-1);
                startpoint = (k-1)*localWidth;
            }
            else {
                localWidth = width-(num_process-2)*(width/(num_process-1));
                startpoint = width-localWidth;
            }
            MPI_Send(&localWidth, 1, MPI_INT, k, k, MPI_COMM_WORLD);
            MPI_Send(&startpoint, 1, MPI_INT, k, k, MPI_COMM_WORLD);
        }
        
        for (int k=0; k<width; k++) {
            MPI_Recv(receivedArray, height+1, MPI_INT, MPI_ANY_SOURCE, 88, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int l=0; l<height; l++) {
                totalArray[receivedArray[height]*height+l] = receivedArray[l];
            }
        }
    } 

    timeEnd = MPI_Wtime();


    printf("the time of MPI_static is %f.\n", (timeEnd-timeStart));

    
    if (enable_X == 1) {
        Display *display;
        Window window;
        int screen;

        //open connection with the server
        display = XOpenDisplay(NULL);
        if (display == NULL) {
            fprintf(stderr, "cannot open display\n");
        }
        
        screen = DefaultScreen(display);

        //set window position
        int x=0;
        int y=0;

        //border width in pixels
        int border_width=0;

        //creat window
        window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width, 
                        BlackPixel(display, screen), WhitePixel(display, screen));


        //create graph
        GC gc;
        XGCValues values;
        long valuemask = 0;

        gc = XCreateGC(display, window, valuemask, &values);
        //XSetBackground(display, gc, Whitepixel(display, screen));
        XSetForeground(display, gc, BlackPixel(display, screen));
        XSetBackground(display, gc, 0X0000FF00);
        XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);

        //map(show) the window
        XMapWindow(display, window);
        XSync(display, 0);

        for (int k=0; k<width; k++) {
            for (int l=0; l<height; l++) {
                XSetForeground (display, gc, 1024*1024*(totalArray[k*height+l]%256));
                XDrawPoint (display, window, gc, k, l);
            }
        }
        
        XFlush(display);
        sleep(5);
    }
} 

int main(int argc, char *argv[])
{
    int rc;

    int rank, num_process;

    if (argc != 9) {
        printf("not enough argument\n");
        exit(0);
    }

    rc = MPI_Init(&argc, &argv);

    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	
    double left_R_range = atof (argv[2]);
    double right_R_range = atof (argv[3]);
    double low_I_range = atof (argv[4]);
    double up_I_range = atof (argv[5]);
    int width = atoi (argv[6]);
    int height = atoi (argv[7]);
    int enable_X = 0;
    if (strcmp (argv[8], "enable") == 0)
        enable_X = 1;
    else 
        enable_X = 0;


    double scale_real = (double)(right_R_range-left_R_range) / width;
    double scale_imag = (double)(up_I_range-low_I_range) / height;
    
    if (width < num_process-1) {
        if (rank == 0) {
           master(left_R_range, low_I_range, scale_real, scale_imag, width, height, num_process, enable_X); 
        }
    }
    else {
        if (rank == 0) {
           master(left_R_range, low_I_range, scale_real, scale_imag, width, height, num_process, enable_X); 
        }
        else {
            slave(rank, height, scale_real, scale_imag, left_R_range, low_I_range);
        }
    }
    
    MPI_Finalize();
    exit(0);
}
