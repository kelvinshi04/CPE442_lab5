#include <iostream>
#include <stdlib.h>
#include <string>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

#define NUM_THREADS 4

struct args{
    Mat *source;
    Mat *gray;
    Mat *dest;
    int startIndex;
    int endIndex;
    uint8_t *status;
};

void* graySobel(void*);
pthread_barrier_t barrierA, barrierB, barrierC;

int main(int argc, char **argv){
    //ensure number of arguments are correct
    if (argc != 2){
        cout << "Error: Invalid input" << endl;
        return 0;
    }

    string videoPath = argv[1];
    VideoCapture cap(videoPath);

    if (!cap.isOpened()){ 
        perror("Error: Could not open video file.");
        return 0;
    }
    pthread_t threads[NUM_THREADS];
    struct args thr_args[NUM_THREADS];
    Mat src, dest, gray;
    
    uint8_t Mat_init = 0, pthread_init = 0, status = 0;
    pthread_barrier_init(&barrierA, NULL, NUM_THREADS + 1);
    pthread_barrier_init(&barrierB, NULL, NUM_THREADS + 1);
    pthread_barrier_init(&barrierC, NULL, NUM_THREADS);


    while (cap.read(src)){
        flip(src, src, -1);
        if (!Mat_init){
            dest.create(src.rows, src.cols, CV_8UC1);
            gray.create(src.rows, src.cols, CV_8UC1);
            Mat_init = 1;
        }
        if (!pthread_init){
            // create pthread + arguments
            for (int i = 0; i < NUM_THREADS; i++){
                thr_args[i].source = &src;
                thr_args[i].dest = &dest;
                thr_args[i].gray = &gray;
                thr_args[i].startIndex = src.rows*i/NUM_THREADS;
                thr_args[i].endIndex = src.rows*(i+1)/NUM_THREADS;
                thr_args[i].status = &status;
                pthread_create(&threads[i], NULL, graySobel, (void *) &thr_args[i]);
            }
            pthread_init = 1;
        }
        // ready
        status = 1;
        pthread_barrier_wait(&barrierB);
        pthread_barrier_wait(&barrierA);
        imshow("sImage", dest);
        waitKey(1);
    }
    status = 0;
    pthread_barrier_wait(&barrierB);

    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

    return 0;
}

void* graySobel(void *arg){
    while(true){
        pthread_barrier_wait(&barrierB);
        struct args *arguments = static_cast<struct args*>(arg);
        if (!(*(arguments->status))){
            pthread_exit(0);
        }
        // grayscale
        for (int r = arguments->startIndex; r < arguments->endIndex; r++){
            Vec3b *curr = arguments->source->ptr<Vec3b>(r);
            uchar *gCurr = arguments->gray->ptr<uchar>(r);
            for (int c = 0; c < arguments->source->cols; c++){
                gCurr[c] = curr[c][0]*0.0722 + curr[c][1]*0.7152 + curr[c][2]*0.2126;
            }
        }
        pthread_barrier_wait(&barrierC);

        // sobel filter
        int16_t xTotal, yTotal, g11, g12, g13, g21, g23, g31, g32, g33, total;
        for (int r = arguments->startIndex; r < arguments->endIndex; r++){
            uchar *tRow, *mRow, *bRow, *sRow;
            if ((r-1) < 0)
                tRow = NULL;
            else 
                tRow = arguments->gray->ptr<uchar>(r-1);
            
            if ((r+1) > arguments->gray->rows-1)
                bRow = NULL;
            else 
                bRow = arguments->gray->ptr<uchar>(r+1);
            mRow = arguments->gray->ptr<uchar>(r);
            sRow = arguments->dest->ptr<uchar>(r);

            for (int c = 0; c < arguments->source->cols; c++){
                // just perform grayscale on edge cases
                if (tRow == NULL || bRow == NULL || c == 0 || c == arguments->source->cols-1){
                    continue;
                }
                else{
                    g11 = tRow[c-1];
                    g12 = tRow[c];
                    g13 = tRow[c+1];
                    g21 = mRow[c-1];
                    g23 = mRow[c+1];
                    g31 = bRow[c-1];
                    g32 = bRow[c];
                    g33 = bRow[c+1];

                    xTotal = -g11 + g13 - g21*2 + g23*2 - g31 + g33;
                    yTotal = g11 + g12*2 + g13 - g31 - g32*2 - g33;

                    total = abs(xTotal) + abs(yTotal);
                    total = (total > 255) ? 255 : ((total < 0) ? 0 : total);
                    sRow[c] = total;
                }
            }
        }
        pthread_barrier_wait(&barrierA);
    }
}
