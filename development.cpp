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
    uint16x8_t ;
    while(true){
        pthread_barrier_wait(&barrierB);
        struct args *arguments = static_cast<struct args*>(arg);
        if (!(*(arguments->status))){
            pthread_exit(0);
        }

        vector<Mat> channels;
        split(*(arguments->source), channels)
        uint8_t *rowB, *rowG, *rowR;
        uint8x8_t bByte, rByte, gByte, grayByte;
        uint16x8_t bU16, rU16, gU16;
        float32x4_t bUp32f, gUp32f, rUp32f, bLo32f, gLo32f, rLo32f;
        uint32x4_t bUp32u, gUp32u, rUp32u, bLo32u, gLo32u, rLo32u;

        // grayscale
        for (int r = arguments->startIndex; r < arguments->endIndex; r++){ 
            rowB = channels[0].ptr<uint8_t>(r);
            rowG = channels[1].ptr<uint8_t>(r);
            rowR = channels[2].ptr<uint8_t>(r);
            uchar *gCurr = arguments->gray->ptr<uchar>(r);
            for (int c = 0; c < arguments->source->cols; c+=8){
                if (arguments->source->cols - c < 8){
                    bByte = vld1_u8(__transfersize(arguments->source->cols - c) rowB);
                    rByte = vld1_u8(__transfersize(arguments->source->cols - c) rowR);
                    gByte = vld1_u8(__transfersize(arguments->source->cols - c) rowG);
                }
                else{
                    // get bytes
                    bByte = vld1_u8(__transfersize(8) rowB);
                    rByte = vld1_u8(__transfersize(8) rowR);
                    gByte = vld1_u8(__transfersize(8) rowG);
                }
                // change size to uint16x8_t
                bU16 = vmovl_u8(bByte);
                gU16 = vmovl_u8(gByte);
                rU16 = vmovl_u8(rByte);

                // split and convert to 2 vectors of 32x4 for floating point compatibility
                bUp32f = vcvtq_f32_u32(vget_high_u16(bByte));
                bLo32f = vcvtq_f32_u32(vget_low_u16(bByte));
                gUp32f = vcvtq_f32_u32(vget_high_u16(gByte));
                gLo32f = vcvtq_f32_u32(vget_low_u16(gByte));
                rUp32f = vcvtq_f32_u32(vget_high_u16(rByte));
                rLo32f = vcvtq_f32_u32(vget_low_u16(rByte));

                // multiply by floating point
                bUp32f = vmulq_n_f32(bUp32f, 0.0722f);
                bLo32f = vmulq_n_f32(bLo32f, 0.0722f);
                gUp32f = vmulq_n_f32(gUp32f, 0.7152f);
                gLo32f = vmulq_n_f32(gLo32f, 0.7152f);
                rUp32f = vmulq_n_f32(rUp32f, 0.2126f);
                rLo32f = vmulq_n_f32(rLo32f, 0.2126f);

                // convert back to unsigned int
                bUp32u = vcvtq_u32_f32(bUp32f);
                bLo32u = vcvtq_u32_f32(bLo32f);
                gUp32u = vcvtq_u32_f32(gUp32f);
                gLo32u = vcvtq_u32_f32(gLo32f);
                rUp32u = vcvtq_u32_f32(rUp32f);
                rLo32u = vcvtq_u32_f32(rLo32f);

                // combine back to 16x8 vector
                bU16 = vcombine_u16(vqmovn_u32(bUp32u), vqmovn_u32(bLo32u));
                gU16 = vcombine_u16(vqmovn_u32(gUp32u), vqmovn_u32(gLo32u));
                rU16 = vcombine_u16(vqmovn_u32(rUp32u), vqmovn_u32(rLo32u));

                // convert to original 8x8 vector format
                bByte = vqmovn_u16(bU16);
                gByte = vqmovn_u16(gU16);
                rByte = vqmovn_u16(rU16);

                // add up values + store
                grayByte = vadd_s8(vadd_s8(bByte, gByte), rByte);
                if (arguments->source->cols - c < 8){
                    vst1_u8(__transfersize(arguments->source->cols - c) gCurr+c, grayByte);
                }    
                else{
                    vst1_u8(__transfersize(8) gCurr+c, grayByte);
                }
            }
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
