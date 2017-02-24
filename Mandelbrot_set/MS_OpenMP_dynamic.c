//MS_OpenMP_dynamic

#include <X11/Xlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>


typedef struct complextype
{
	double real, imag;
} Compl;


int main(int argc, char* argv[])
{
    
    if (argc != 9) {
        printf("hot enough argument!\n");
        exit(0);
    }
    
    int thread_numbers = atoi (argv[1]);
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

    double scale_real = (double) (right_R_range-left_R_range) / width;
    double scale_imag = (double) (up_I_range-low_I_range) / height;

    int* pixelArray = malloc ((width*height) * sizeof(int));
    Compl z, c;
    int repeats;
    double temp, lengthsq;
    int i, j;
    double timeStart, timeEnd;

    timeStart = omp_get_wtime();
    #pragma omp parallel num_threads(thread_numbers) shared(scale_real, scale_imag, pixelArray) private(i,j,z,c,temp, lengthsq, repeats)
    {
        #pragma omp for schedule(dynamic) collapse(2) nowait
        for(i=0; i<width; i++) {
            for(j=0; j<height; j++) {
                z.real = 0.0;
                z.imag = 0.0;
                c.real = left_R_range + ((double) i * scale_real); /* Theorem : If c belongs to M(Mandelbrot set), then |c| <= 2 */
                c.imag = low_I_range + ((double) j * scale_imag); /* So needs to scale the window */
                repeats = 0;
                lengthsq = 0.0;

                while(repeats < 100000 && lengthsq < 4.0) { /* Theorem : If c belongs to M, then |Zn| <= 2. So Zn^2 <= 4 */
                    temp = z.real*z.real - z.imag*z.imag + c.real;
                    z.imag = 2*z.real*z.imag + c.imag;
                    z.real = temp;
                    lengthsq = z.real*z.real + z.imag*z.imag; 
                    repeats++;
                }
                pixelArray[i*height + j] = repeats;

            }
        }
    }
    timeEnd = omp_get_wtime();

    printf("the time for OpenMP_dynamic is %f.\n", (timeEnd-timeStart));
    
    if (enable_X == 1) {
        Display *display;
        Window window;      //initialization for a window
        int screen;         //which screen 

        /* open connection with the server */ 
        display = XOpenDisplay(NULL);
        if(display == NULL) {
            fprintf(stderr, "cannot open display\n");
            return 0;
        }

        screen = DefaultScreen(display);

        
        /* set window position */
        int x = 0;
        int y = 0;

        /* border width in pixels */
        int border_width = 0;

        /* create window */
        window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
                        BlackPixel(display, screen), WhitePixel(display, screen));
        
        /* create graph */
        GC gc;
        XGCValues values;
        long valuemask = 0;
        
        gc = XCreateGC(display, window, valuemask, &values);
        //XSetBackground (display, gc, WhitePixel (display, screen));
        XSetForeground (display, gc, BlackPixel (display, screen));
        XSetBackground(display, gc, 0X0000FF00);
        XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
        
        /* map(show) the window */
        XMapWindow(display, window);
        XSync(display, 0);
        
        /* draw points */
        for (int k=0; k<width; k++) {
            for (int l=0; l<height; l++) {
                XSetForeground (display, gc, 1024 * 1024 * (pixelArray[k*height+l] % 256));
                XDrawPoint (display, window, gc, k, l);
            }
        }
        XFlush(display);
        sleep(5);
    }
	exit(0);
}
