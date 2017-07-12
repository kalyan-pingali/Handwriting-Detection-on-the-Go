#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <semaphore.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>


using namespace cv;
using namespace std;


// global defintions
#define NSEC_PER_SEC (1000000000)
#define ERROR (-1)
#define OK (0)


//Resolution Definitions
#define HRES 640
#define VRES 480


//Window defintions
char Live_window_name[] = "Live Capture";


//Scheduler defintions
int rt_max_prio, rt_min_prio;


// thread defintions
pthread_t live_video_thread, frame_grabber_thread, print_it_thread;
pthread_attr_t live_video_thread_attr, frame_grabber_thread_attr, print_it_thread_attr;
struct sched_param main_param, live_video_thread_param, frame_grabber_thread_param, print_it_thread_param;


// defining number_of_frames, total_time for evaulating average execution time
double number_of_frames=0;
double total_time=0;
int dev = 0;

	
// semaphone defintions
sem_t sem_frame_grabber, sem_live_video, sem_live_video_ack, sem_print_it;


//opencv defintions
CvCapture* capture;
IplImage* frame, grabbed_frame;


// time stamping variable definitions
static struct timespec frame_grabber_rtclk_dt = {0, 0};
static struct timespec frame_grabber_rtclk_start_time = {0, 0};
static struct timespec frame_grabber_rtclk_stop_time = {0, 0};
static struct timespec live_video_rtclk_dt = {0, 0};
static struct timespec live_video_rtclk_start_time = {0, 0};
static struct timespec live_video_rtclk_stop_time = {0, 0};
static struct timespec print_it_rtclk_dt = {0, 0};
static struct timespec print_it_rtclk_start_time = {0, 0};
static struct timespec print_it_rtclk_stop_time = {0, 0};
static struct timespec live_video_run_time = {0, 199999990}; //Almost 200 milliseconds, leaving time for computation of 2 statements (live video)
static struct timespec live_video_left_time = {0, 0};
static struct timespec remaining_time = {0, 0};
static struct timespec live_video_elapsed_dt = {0, 0};
static struct timespec live_video_rtclk_int_time = {0, 0};
static struct timespec init_time = {0, 0};
static struct timespec to_frame_grab_time = {0, 0};
static struct timespec adjustment_time = {0, 160000000};



//Frame grabber switch counter definitions
int switch_counter=0;


//Mat definitions (Frame grabbed, Gray scale version of frame grabbed, bounding reactangle of frame grabbed )
Mat mat_frame_grabber, gray_image, bounding_rectangle;
Mat previous_frame = Mat::zeros(VRES,HRES,CV_8U);
Mat current_frame = Mat::zeros(VRES,HRES,CV_8U);


//kNN implementation definitions
int K=5;
int train_samples = 150;			// number samples
int train_class = 26; 			// A ~ Z , alphabets
Rect rect(0, 0, 40, 40);
Mat mtrain_class = Mat(train_samples*train_class,1, CV_32F);			// Preparing a Mat to store identifiers of each corresponding image in train_data
Mat train_data = Mat(train_samples*train_class,rect.area(), CV_32F);	// Preparing a Mat that stores pixel data from each training image as a row
CvKNearest knn;
float response;

//For setting affinity!
cpu_set_t cpuset; 


//Thinning algorithm for obtaining pixel width lines [Reference:http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/]
void thinningIteration(Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    Mat marker = Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;    // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    // south (pBelow)

    uchar *pDst;

    // initialize row pointers
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (y = 1; y < img.rows-1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = img.ptr<uchar>(y+1);

        pDst = marker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols-1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);

            int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) + 
                     (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) + 
                     (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                     (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }

    img &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 * 		src  The source image, binary with range = [0,255]
 * 		dst  The destination image
 */
void thinning(const Mat& src, Mat& dst)
{
    dst = src.clone();
    dst /= 255;         // convert to binary image
    
    Mat prev = Mat::zeros(dst.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } 
    while (countNonZero(diff) > 0);

    dst *= 255;
}






// Print Scheduler function to print current scheduling policy
void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
     case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
     case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n");
           break;
     case SCHED_RR:
           printf("Pthread Policy is SCHED_RR\n");
           break;
     default:
       printf("Pthread Policy is UNKNOWN\n");
   }
}





// delta_t as in posix_clock (calculates the time difference between time stamps)
int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec=stop->tv_sec - start->tv_sec;
  int dt_nsec=stop->tv_nsec - start->tv_nsec;

  if(dt_sec >= 0)
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }
  else
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }

  return(OK);
}





void *frame_grabber(void *someinput)
{
	while(1)
	{
		sem_wait(&sem_frame_grabber);
	    //if(mat_frame_grabber
		//Mat source1 = imread("test.png");
		//cvtColor(source1,source1,CV_RGB2GRAY);
		clock_gettime(CLOCK_REALTIME, &frame_grabber_rtclk_start_time);
		delta_t(&frame_grabber_rtclk_start_time, &adjustment_time, &to_frame_grab_time);
		delta_t(&to_frame_grab_time, &init_time, &to_frame_grab_time);
		cvtColor(mat_frame_grabber, gray_image, CV_RGB2GRAY);
	    namedWindow("Gray Image", WINDOW_AUTOSIZE);
		blur( gray_image, gray_image, Size(2,2) );
		
		//printf("Reached post gray image print\n");
		//usleep(200000);
		vector<vector<Point> > contours;
		Mat threshold_output;
		vector<Vec4i> hierarchy;
		
		int i;
		
		threshold( gray_image, threshold_output, 100, 255, THRESH_BINARY );
		//imshow("theshold output", threshold_output);
		threshold_output.copyTo(current_frame);
		absdiff(current_frame,previous_frame,threshold_output);
		//printf("\n\n\n\n\n\ndepth of threshold image is : %d\n\n\n\n\n\n\n",threshold_output.depth());
		//gray_image.convertTo(gray_image, CV_32F);
		findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		//printf("Contour size : %d\n", contours.size());
		
		//Largest Area intialized to 0
		//Largest contour index set to default : 0
		int largest_area=0;
		int largest_contour_index=0;
		Rect bounding_rectangle;
		//printf("\n1\n");
		for( i = 0; i< contours.size(); i++ ) // iterate through each contour.
		{
        double area = contourArea( contours[i] );  // Find the area of contour

			if( area > largest_area )
			{
				if( area < 70000)				// 70000 to prevent complete frame to get detected as a contour (happens due to lighting)
				{
				largest_area = area;
				largest_contour_index = i;               // Store the index of largest contour
				bounding_rectangle = boundingRect( contours[i] ); // Find the bounding rectangle for biggest contour
				}
				else
				{
					//printf("\n\n\n\n\n\nDefault contour detected\n\n AREA IS %f\n\n\n\n",area);
				}
			}
		}

		//vector<vector<Point> > contours_poly( contours.size() );
		
		//for( i = 0; i< contours.size(); i++ )
		//{
		//	printf("Loop number : %d\n", i);
		//approxPolyDP( Mat(contours[largest_contour_index]), contours_poly[largest_contour_index], 3, true );
		//bounding_rectangle = boundingRect( Mat(contours_poly[largest_contour_index]) );
		//	printf("Size of this contour is: %lf\n", contourArea(contours[largest_contour_index]));
		//}
		//for( i = 0; i< contours.size(); i++ )
		//{
		//printf("2\n");
		
		Mat cropped_image = gray_image(bounding_rectangle);
		
		//cropped_image.convertTo(cropped_image, CV_32F);
		//Drawing the bounding rectangle onto the gray image	
			
		rectangle(threshold_output, bounding_rectangle.tl(), bounding_rectangle.br(), Scalar(0,0,0), 2, 8, 0);
		
		//}
		//printf("3\n");
		
		//imshow("Gray Image", gray_image);
		Mat store_image=Mat::zeros(40,40,CV_32F); //CV_8U => gray scale image
				
		//printf("4\n");
		
		resize(cropped_image,store_image,store_image.size(), 0, 0, INTER_AREA);
		bitwise_not(store_image, store_image);
			
		//imshow("inverted image", cropped_image);
		//thinning(store_image, store_image);
		//store_image.convertTo(store_image, CV_32F);
		
		Mat onedimmat = store_image.clone().reshape(0,1);
		
		//printf("5\n");
		//onedimmat.convertTo(onedimmat, CV_32F);
		//printf("reached till before knn finder");
		//result = knn.find_nearest(onedimmat, K);
		//printf("knn should have ran?");
		//printf("Result is : %d",result);

		namedWindow("Store Image", WINDOW_AUTOSIZE);
		imshow("Store Image", store_image);
		
		//Mat onedimmat = source1.clone().reshape(0,1);
		/*printf("\ndepth: %d\n",onedimmat.depth());
		printf("channels: %d\n",onedimmat.channels());*/
		//cvtColor(onedimmat,onedimmat,CV_RGB2GRAY);
		/*printf("depth: %d\n",onedimmat.depth());
		printf("channels: %d\n",onedimmat.channels());
		printf("\n\n\n\n\nendonsjvnelwvsnlsjdnljsndlvnlaevdnlvn\n");*/
		
		onedimmat.convertTo(onedimmat, CV_32F);
		
		/*printf("depth: %d\n",onedimmat.depth());
		printf("channels: %d\n",onedimmat.channels());*/
		
		Mat results;
		Mat nR,dists;
		
		//Applying kNN on  onedimmat to get output
		
		response = knn.find_nearest(onedimmat, K,results,nR,dists);
		
		//Debugging to figure out where response is stored from knn.find_nearest()
		
		/*printf("\n\n\n\nwidth: %d; height: %d\n\n\n\n",results.rows,results.cols);
		printf("results is:  ");
		cout<<results;
		//printf("\n\n\n\n");
		printf("\n\n\n\n\nDetected alphabet is : %c\n\n\n\n\n",results.at<int>(0));
		
		printf("\n\n\n\nwidth of nR: %d; height: %d\n\n\n\n",nR.rows,nR.cols);
		printf("nR is:  ");
		cout<<nR;
		printf("\n\n\n\n");
		
		printf("\n\n\n\nwidth of dists: %d; height: %d\n\n\n\n",dists.rows,dists.cols);
		printf("results is:  ");
		cout<<dists;
		printf("\n\n\n\n");*/
		
		int answer;
		answer = int(response);
		//printf("knn should have ran?");
		printf("\n\n\nResult is : %c\n\n\n",answer);
		char something[2] = {'\0'};
		sprintf(something,"%c",answer);
		Point textOrg(250,250);
		double scalee = 10;
		putText(gray_image, something, textOrg, FONT_HERSHEY_COMPLEX, scalee, Scalar(255,0,0));
		
		imshow("Gray Image", gray_image);
		current_frame.copyTo(previous_frame);
		clock_gettime(CLOCK_REALTIME, &frame_grabber_rtclk_stop_time);
		delta_t(&frame_grabber_rtclk_stop_time, &frame_grabber_rtclk_start_time, &frame_grabber_rtclk_dt);
		sem_post(&sem_print_it);
		//cvWaitKey(0);
	}
}





void *live_video(void *someinput)
{
	while(1)
    {
		sem_wait(&sem_live_video);
		clock_gettime(CLOCK_REALTIME, &live_video_rtclk_start_time);
		

		frame=cvQueryFrame(capture);
		
		if(!frame) break;
			
		cvShowImage( Live_window_name, frame );
		
		//printf("RT clock DT seconds = %ld, milliseconds = %ld\n", 
		//rtclk_dt.tv_sec, rtclk_dt.tv_nsec/1000000);
			
		
		if(switch_counter==15)
		{
			Mat mat_frame(frame);
			mat_frame.copyTo(mat_frame_grabber);
			switch_counter=0;
			sem_post(&sem_frame_grabber);
		}
		switch_counter+=1;
		char c = cvWaitKey(2);
		if( c == 27 ) break;
				
		number_of_frames+=1;
		clock_gettime(CLOCK_REALTIME, &live_video_rtclk_int_time);
		delta_t(&live_video_rtclk_int_time, &live_video_rtclk_start_time, &live_video_elapsed_dt);
		delta_t(&live_video_run_time, &live_video_elapsed_dt, &live_video_left_time);
		nanosleep(&live_video_left_time, &remaining_time);
		clock_gettime(CLOCK_REALTIME, &live_video_rtclk_stop_time);
		delta_t(&live_video_rtclk_stop_time, &live_video_rtclk_start_time, &live_video_rtclk_dt);
		total_time+=live_video_rtclk_dt.tv_nsec/1000000;
		sem_post(&sem_live_video_ack);
    }
	
	
}





void *print_it(void *someinput)
{
	while(1)
    {
		sem_wait(&sem_print_it);
		clock_gettime(CLOCK_REALTIME, &print_it_rtclk_start_time);
		printf("Deadline of live_video thread is : seconds = %ld, milliseconds = %ld\n", 
        live_video_rtclk_dt.tv_sec, live_video_rtclk_dt.tv_nsec/1000000);
	    printf("Total time taken by frame_grabber thread is : seconds = %ld, milliseconds = %ld\n", 
        frame_grabber_rtclk_dt.tv_sec, frame_grabber_rtclk_dt.tv_nsec/1000000);
		printf("frame_grabber thread spawned at : seconds = %ld, milliseconds = %ld\n", 
        to_frame_grab_time.tv_sec, to_frame_grab_time.tv_nsec/1000000);
		printf("Deadline of live_video thread is : seconds = %ld, milliseconds = %ld\n", 
        live_video_elapsed_dt.tv_sec, live_video_elapsed_dt.tv_nsec/1000000);
		usleep(1000);	//To ensure this thread has minimum run-time
		//printf("Average time for live_video thread is : %f\n",total_time/number_of_frames);
		clock_gettime(CLOCK_REALTIME, &print_it_rtclk_stop_time);
		delta_t(&print_it_rtclk_stop_time, &print_it_rtclk_start_time, &print_it_rtclk_dt);
		printf("Total time taken by print_it thread is : seconds = %ld, milliseconds = %ld\n", 
        print_it_rtclk_dt.tv_sec, print_it_rtclk_dt.tv_nsec/1000000);
    }
	
	
}





void knn_init(void)
{
	system("ls ./Output > imagenames.txt");	// Stores all image names from train set (available in directory 'Output') into imagenames.txt
	FILE *stream;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    stream = fopen("./imagenames.txt", "r");
    if (stream == NULL)
        exit(1);
	int count = 0;
	int alpha_ascii = 65;
	int next_alphabet = 55;
	
	while ((read = getline(&line, &len, stream)) != -1)
	{
		int line_length = strlen(line);
		char line_ret[25] = {'\0'};
		strcat(line_ret, "./Output/");
		line[line_length-1]='\0';		//Replacing \n with \0 to make a valid filename
		strcat(line_ret, line);
		//printf("%s", line_ret);
		Mat src = imread(line_ret);
		//src = src.reshape(0);		//Changing into continous!
		//cout<<src.channels(); //Debug statement
		Mat onemat = src.clone().reshape(0,1);
		
		//printf("\ndepth: %d\n",onemat.depth());
		//printf("channels: %d\n",onemat.channels());
		//printf("\n\n\n\nCloneeeeeeeeee done\n\n\n\n");
		//cout<<onemat.channels(); //Debug statement
		
		
		cvtColor(onemat, onemat, CV_RGB2GRAY);
		//printf("\ndepth: %d\n",onemat.depth());
		//printf("channels: %d\n",onemat.channels());
	
		//cout<<onemat.channels(); //Debug statement
		//cout<<train_data.channels(); //Debug statement
		//scout<<train_data.channels();
		//printf("convert to works fine\n\n");
		//onemat.convertTo(onemat, CV_32FC1);
		//to copy a row into another mat
		//cout<<onemat.channels();
		onemat.copyTo(train_data.row(count));// = (onemat.row(count) + 0);
		//printf("convert and copied works fine\n\n");
		//printf("Looped %d times\n",count);
		mtrain_class.at<float>(count) = (float)alpha_ascii;
		//printf("Ascii value stored: %d ;  %s\n", alpha_ascii,line_ret);
		//printf("mtrain: %f\n",mtrain_class.at<float>(count));
		count+=1;
		//mtrain_class[count] = alpha_ascii;
		if((count)%next_alphabet == 0)
		{
			alpha_ascii += 1;
			//break;
		}
	}
	//if(train_data.depth()==CV_32FC1)
	//{
	//	cout<<" train_data IS 32F data type";
	//}
	
	
	//printf("Value (should be 66) : %f", mtrain_class.at<float>(20));
	//printf("Value (should be 67) : %f", mtrain_class.at<float>(30));
	//printf("Value (should be 68) : %f", mtrain_class.at<float>(40));
	//printf("valuesvkjns of mtrain : %d",mtrain_class.depth());
	//train_data.convertTo(train_data, CV_32FC1);
	//mtrain_class.convertTo(mtrain_class, CV_32FC1);
	
	//imshow("traindata",train_data);
	//cvWaitKey(0);
	//imshow("mtrain_class",mtrain_class);
	//cvWaitKey(0);
	//if(train_data.depth()==CV_32FC1)
	//{
	//	cout<<" train_data IS 32F data type";
	//}
	//if(mtrain_class.depth()==CV_32FC1)
	//{
	//	cout<<" mtrain_class IS 32F data type";
	//}
	
	
	knn.train(train_data, mtrain_class);
	//printf("Value at the 20th instance in mtrain_class (should be 66) : %f", mtrain_class.at<float>(20));
	//printf("Value at the 30th instance in mtrain_class (should be 67) : %f", mtrain_class.at<float>(30));
	//printf("Value at the 40th instance in mtrain_class (should be 68) : %f", mtrain_class.at<float>(40));
	
	
	Mat source1 = imread("test.png");
	//cvtColor(source1,source1,CV_RGB2GRAY);
	//Mat store_image=Mat::zeros(40,40,CV_32F); //CV_8U => gray scale image
	//resize(source1,store_image,store_image.size(), 0, 0, INTER_AREA);
	//Mat onedimmat=Mat::zeros(1600,1,CV_32F);
	Mat onedimmat = source1.clone().reshape(0,1);
	//printf("\ndepth: %d\n",onedimmat.depth());
	//printf("channels: %d\n",onedimmat.channels());
	cvtColor(onedimmat,onedimmat,CV_RGB2GRAY);
	//printf("depth: %d\n",onedimmat.depth());
	//printf("channels: %d\n",onedimmat.channels());
	//printf("\n\n\n\n\nendonsjvnelwvsnlsjdnljsndlvnlaevdnlvn\n");
	onedimmat.convertTo(onedimmat, CV_32F);
	//printf("depth: %d\n",onedimmat.depth());
	//printf("channels: %d\n",onedimmat.channels());
	Mat results;
	Mat nR,dists;
	
	
	
	response = knn.find_nearest(onedimmat, K,results,nR,dists);
	/*printf("\n\n\n\nwidth: %d; height: %d\n\n\n\n",results.rows,results.cols);
	printf("results is:  ");
	cout<<results;
	//printf("\n\n\n\n");
	printf("\n\n\n\n\nDetected alphabet is : %d\n\n\n\n\n",results.at<int>(0));
	
	
	printf("\n\n\n\nwidth of nR: %d; height: %d\n\n\n\n",nR.rows,nR.cols);
	printf("nR is:  ");
	cout<<nR;
	printf("\n\n\n\n");
	
	printf("\n\n\n\nwidth of dists: %d; height: %d\n\n\n\n",dists.rows,dists.cols);
	printf("results is:  ");
	cout<<dists;
	printf("\n\n\n\n");*/
	
	int answer;
	answer = int(response);
	//printf("knn should have ran?");
	printf("\n\n\nResponse for test image (input is the alphabet 'I') is : %c\n\n\n",answer);
		
		
	usleep(1000000);
}





int main( int argc, char** argv )
{	
	CPU_ZERO(&cpuset);       //clears the cpuset
    CPU_SET( 2 , &cpuset);   //set CPU 2 on cpuset

	int rc;
    printf("Before adjustments to scheduling policy:\n");
    print_scheduler();
   
    // Updating scheduler to SCHED_FIFO
    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    main_param.sched_priority = rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    printf("After adjustments to scheduling policy:\n");
    print_scheduler();
 
	//Initialization of semaphores
	sem_init(&sem_live_video,0,0);
	sem_init(&sem_frame_grabber,0,0);
	sem_init(&sem_live_video_ack,0,0);
	sem_init(&sem_print_it,0,0);
	
	//Live Video Thread Initialization
	pthread_attr_init(&live_video_thread_attr);
    pthread_attr_setinheritsched(&live_video_thread_attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&live_video_thread_attr, SCHED_FIFO);

    live_video_thread_param.sched_priority = rt_max_prio;
    pthread_attr_setschedparam(&live_video_thread_attr, &live_video_thread_param);
 
 
	//Frame Grabber Thread Initialization
    pthread_attr_init(&frame_grabber_thread_attr);
    pthread_attr_setinheritsched(&frame_grabber_thread_attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&frame_grabber_thread_attr, SCHED_FIFO);
    
    frame_grabber_thread_param.sched_priority = rt_max_prio;
    pthread_attr_setschedparam(&frame_grabber_thread_attr, &frame_grabber_thread_param);
    

	//Print it Thread Initialization
    pthread_attr_init(&print_it_thread_attr);
    pthread_attr_setinheritsched(&print_it_thread_attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&print_it_thread_attr, SCHED_FIFO);
    
    print_it_thread_param.sched_priority = rt_max_prio;
    pthread_attr_setschedparam(&print_it_thread_attr, &print_it_thread_param);

    
	//KNN Intialization (Forming data sets)
	knn_init();
	
	printf("kNN initialization successful\n");
	
    // Creating windows
    namedWindow( Live_window_name, CV_WINDOW_AUTOSIZE );
    
	//Defining capture properties
    capture = (CvCapture *)cvCreateCameraCapture(dev);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, HRES);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, VRES);
   
   
    if(pthread_create(&live_video_thread, &live_video_thread_attr, live_video, NULL))
    {
      printf("Live video thread creation failed\n");
    }
     else
	printf("Live video thread creation successful\n");

	if(pthread_create(&frame_grabber_thread, &frame_grabber_thread_attr, frame_grabber, NULL))
    {
      printf("Frame grabber thread creation failed\n");
    }
     else
	printf("Frame grabber thread creation successful\n");

	if(pthread_create(&print_it_thread, &print_it_thread_attr, print_it, NULL))
    {
      printf("Print it thread creation failed\n");
    }
     else
	printf("Print it thread creation successful\n");

	clock_gettime(CLOCK_REALTIME, &init_time);
	//delta_t(&init_time, &adjustment_time, &init_time);

	while(1)
	{
		
		sem_post(&sem_live_video);
		sem_wait(&sem_live_video_ack);
	
	}
	
	pthread_setaffinity_np(live_video_thread, sizeof(cpu_set_t), &cpuset);
	pthread_setaffinity_np(frame_grabber_thread, sizeof(cpu_set_t), &cpuset);
	pthread_setaffinity_np(print_it_thread, sizeof(cpu_set_t), &cpuset);
	
	pthread_join(live_video_thread, NULL);
	pthread_join(frame_grabber_thread, NULL);
	pthread_join(print_it_thread, NULL);
	//printf("Time per frame capture (fps on average) : %lf milliseconds\n", total_time/number_of_frames);
	
}


// Calibration for background elimination! (Pink and Red colors to be used for writing)

// H min = 0 
// S min = 0
// V min = 66

// H max = 179
// S max = 116
// V max = 255