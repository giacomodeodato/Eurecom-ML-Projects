#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>


//documentation https://docs.opencv.org/2.4/doc/tutorials/imgproc/table_of_content_imgproc/table_of_content_imgproc.html
using namespace std;
using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

char* window_name0 = "Original Image";
char* window_name1 = "Grayscale Image";
char* window_name2 = "Image After Histogram Equalization";
char* remap_window1 = "Remap - upside down";
char* remap_window2 = "Remap - reflection in the x direction";
char* window_name4 = "Median Filtered Image";
char* window_name5 = "Gaussian Filtered Image";

int ddepth = CV_16S;
int scale = 1;
int delta = 0;
int kernel_size = 3;
char* window_name6 = "Laplace Demo";

char* window_name7 = "Sobel Demo - Simple Edge Detector";

/// Global Variables
int MAX_KERNEL_LENGTH = 6;


/** @function main */
int main( int argc, char** argv )
{

    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

	// Call the appropriate function in OPENCV to load the image
	src = imread( argv[1] );
	if( !src.data )
	{ return -1; }

	// Create a window called "Original Image" and show original image
	namedWindow( window_name0, CV_WINDOW_AUTOSIZE );
	imshow( window_name0, src );

	// Call the appropriate function in OPENCV to convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY); //src_gray=imread( argv[1],0 );

	// Create a window called "Grayscale Image" and show grayscale image
	namedWindow( window_name1, CV_WINDOW_AUTOSIZE );
	imshow( window_name1, src_gray );

	// Apply histogram equalization to the grayscale image
	Mat src_eq;
	equalizeHist(src_gray, src_eq);
	
	// Create a window called "Image After Histogram Equalization" and show the image after histogram equalization
	namedWindow( window_name2, CV_WINDOW_AUTOSIZE );
	imshow( window_name2, src_eq );

	// Apply remapping; first turn the image upside down and then reflect the image in the x direction
	// For this part, the upside down image and the flipped left image are created as the Mat variables "image_upsidedown" and "image_flippedleft". Also, map_x and map_y are created with the same size as equalized_image:
	Mat image_upsidedown;
	Mat image_flippedleft;
	
	Mat map_x, map_y;

	image_upsidedown.create( src.size(), src_eq.type() );
	image_flippedleft.create( src.size(), src_eq.type() );

	map_x.create( src.size(), CV_32FC1 );
	map_y.create( src.size(), CV_32FC1 );
	
	
	// Apply upside down operation to the image for which histogram equalization is applied.
	for( int j = 0; j < src_eq.rows; j++ )
	{
		for( int i = 0; i < src_eq.cols; i++ )
		{
			map_x.at<float>(j,i) = i ;
			map_y.at<float>(j,i) = src_eq.rows - j ;
		}
	}
	remap( src_eq, image_upsidedown, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );

	// Create a window called "Remap - upside down" and show the image after applying remapping - upside down
	namedWindow( remap_window1, CV_WINDOW_AUTOSIZE );
	imshow( remap_window1, image_upsidedown );


	// Apply reflection in the x direction operation to the image for which histogram equalization is applied.
	for( int j = 0; j < src_eq.rows; j++ )
	{
		for( int i = 0; i < src_eq.cols; i++ )
		{
			map_x.at<float>(j,i) = src_eq.cols - i ;
			map_y.at<float>(j,i) = j ;
		}
	}
	remap( src_eq, image_flippedleft, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
	
	// Create a window called "Remap - reflection in the x direction" and show the image after applying remapping - reflection in the x direction
	
	namedWindow( remap_window2, CV_WINDOW_AUTOSIZE );
	imshow( remap_window2, image_flippedleft );
   
	// Apply Median Filter to the Image for which histogram equalization is applied 
	Mat img_median;
	medianBlur(src_eq, img_median, MAX_KERNEL_LENGTH-1);

	// Create a window called "Median Filtered Image" and show the image after applying median filtering
	namedWindow( window_name4, CV_WINDOW_AUTOSIZE );
	imshow( window_name4, img_median );
	
    // Remove noise from the image for which histogram equalization is applied by blurring with a Gaussian filter
	Mat img_gaussian;
	GaussianBlur(src_eq,img_gaussian, Size(3,3), 0,0);

	// Create a window called "Gaussian Filtered Image" and show the image after applying Gaussian filtering
	namedWindow( window_name5, CV_WINDOW_AUTOSIZE );
	imshow( window_name5, img_gaussian );

	/// Apply Laplace function to compute the edge image using the Laplace Operator
	Mat img_laplacian, img_laplacian_abs;
	Laplacian(img_gaussian,img_laplacian,ddepth,kernel_size,scale,delta);
	convertScaleAbs( img_laplacian, img_laplacian_abs );

    /// Create window called "Laplace Demo" and show the edge image after applying Laplace Operator
	namedWindow( window_name6, CV_WINDOW_AUTOSIZE );
	imshow( window_name6, img_laplacian_abs);


	// Apply Sobel Edge Detection
	/// Appropriate variables grad, grad_x and grad_y, abs_grad_x and abs_grad_y are generated
	Mat grad;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Compute Gradient X
	Sobel( img_gaussian, grad_x, ddepth, 1, 0, kernel_size, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Compute Gradient Y
	Sobel( img_gaussian, grad_y, ddepth, 0, 1, kernel_size, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Compute Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
	
	/// Create window called "Sobel Demo - Simple Edge Detector" and show Sobel edge detected image
	namedWindow( window_name7, CV_WINDOW_AUTOSIZE );
	imshow( window_name7, grad);


	/// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
  }