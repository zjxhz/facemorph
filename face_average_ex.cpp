#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
using namespace std;
using namespace cv;


// Constrains points to be inside boundary
void constrainPoint(Point2f &p, Size sz)
{
    p.x = min(max( (double)p.x, 0.0), (double)(sz.width - 1));
    p.y = min(max( (double)p.y, 0.0), (double)(sz.height - 1));
    
}

void push_boundary_points(vector<Point2f>& points, Size size){
    int w = size.width;
    int h = size.height;
    vector<Point2f> boundaryPts;
    boundaryPts.push_back(Point2f(0,0));
    boundaryPts.push_back(Point2f(w/2, 0));
    boundaryPts.push_back(Point2f(w-1,0));
    boundaryPts.push_back(Point2f(w-1, h/2));
    boundaryPts.push_back(Point2f(w-1, h-1));
    boundaryPts.push_back(Point2f(w/2, h-1));
    boundaryPts.push_back(Point2f(0, h-1));
    boundaryPts.push_back(Point2f(0, h/2));
    for ( size_t j = 0; j < boundaryPts.size(); j++)
    {
        points.push_back(boundaryPts[j]);
    }
}


std::vector<Point2f> getLandmarksPoints(const Size& size, const string& landmarks_path, bool includeBoundaryPoints = true){
    ifstream in(landmarks_path);
    int x, y;
    std::vector<Point2f> points;
    while(in >> x >> y)
    {
        Point2f point(x, y);
        points.push_back(point);
    }
    in.close();
    if(includeBoundaryPoints){
        push_boundary_points(points, size);    
    }
    return points;
}

void drawLandmarks(const string& path,  const string& landmarks_path){
    Mat face = imread(path);
    std::vector<Point2f> points = getLandmarksPoints(face.size(), landmarks_path, false);
    Scalar color(255, 0, 0);
    for( int i = 0; i < points.size(); i++){
        Point2f point = points[i];
        circle(face, point, 3, color, -1);
    }
    imshow("Landmarks", face);
    waitKey(0);
}

std::vector<Vec6f> getDelaunayTriangles(Size size, const string& landmarks_path, bool includeBoundaryPoints = true)
{
    std::vector<Point2f> points = getLandmarksPoints(size, landmarks_path, includeBoundaryPoints);
    Rect rect(0, 0, size.width, size.height);
    Subdiv2D subdiv(rect);
    for(int i = 0; i < points.size(); i++){
        constrainPoint(points[i], size);
        subdiv.insert(points[i]);
    }
    std::vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    return triangleList;
}

vector<vector<int> > getOrderedTriangleListIndexes(Size size, const string& landmarks_path, bool includeBoundaryPoints = true){
    vector<Vec6f> triangleList = getDelaunayTriangles(size, landmarks_path, includeBoundaryPoints);
    vector<Point2f> points = getLandmarksPoints(size, landmarks_path, includeBoundaryPoints);
    Rect rect(0, 0, size.width, size.height);
    vector<vector<int> > orderedTriangleListIndexes;
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        vector<Point2f> pt(3);
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5 ]);
        
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
            vector<int> indexes;
            for(int j = 0; j < 3; j++){
                for(size_t k = 0; k < points.size(); k++){
                    if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1){
                        indexes.push_back(k);
                        break;
                    } 
                }
            }
            orderedTriangleListIndexes.push_back(indexes);
        }
    }
    return orderedTriangleListIndexes;
}

vector<vector<Point2f> > getOrderedDelaunayTriangles(Size size, const string& landmarks_path, 
        const vector<vector<int> >& orderedTriangleListIndexes, bool includeBoundaryPoints = true){
    std::vector<Vec6f> triangleList = getDelaunayTriangles(size, landmarks_path); 
    vector<Point2f> points = getLandmarksPoints(size, landmarks_path, includeBoundaryPoints);
    vector<vector<Point2f> > orderedTriangleList;
    for(int i = 0; i < orderedTriangleListIndexes.size(); i++){
        vector<int> triangleIndexes = orderedTriangleListIndexes[i];
        std::vector<Point2f> triangle;
        for(int j = 0; j < 3; j++){
            int pointIndex = triangleIndexes[j];
            triangle.push_back(points[pointIndex]); 
        }
        orderedTriangleList.push_back(triangle);    
    }
    return orderedTriangleList;
}

vector<vector<Point2f> > getOrderedDelaunayTriangles(Size size, const string& landmarks_path, bool includeBoundaryPoints = true)
{   
    vector<vector<int> > orderedTriangleListIndexes = getOrderedTriangleListIndexes(size, landmarks_path, includeBoundaryPoints);
    return getOrderedDelaunayTriangles(size, landmarks_path, orderedTriangleListIndexes);
}



vector<vector<Point2f> > trianglesToPointsVector(vector<Vec6f> triangles, Size size){
    vector<vector<Point2f> > points;
    for(int i = 0; i < triangles.size(); i++){
        Vec6f t = triangles[i];
        std::vector<Point2f> pt(3);
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
        constrainPoint(pt[0], size);
        constrainPoint(pt[1], size);
        constrainPoint(pt[2], size);    
        points.push_back(pt);
    }
    return points;
}

int min(int a, int b, int c){
    return min(min(a,b), c);
}

int max(int a, int b, int c){
    return max(max(a,b), c);
}

int middle(int a, int b, int c){
    return (min(a, b, c) + max(a, b, c)) / 2;
}

void putIndexInTriangle(Mat& img, const std::vector<Point2f>& points, int index){
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int midX = middle(points[0].x, points[1].x, points[2].x);
    int midY = middle(points[0].y, points[1].y, points[2].y);
    Moments mu = moments(points, false);
    int centerX = mu.m10 / mu.m00 - 10;   
    int centerY = mu.m01 / mu.m00;


    putText(img, to_string(index), Point(centerX, centerY), fontFace, fontScale,
        Scalar::all(255));
}

void drawDelaunayTriangles(Mat& img, const string& landmarks_path){
    std::vector<vector<Point2f> > triangles = getOrderedDelaunayTriangles(img.size(), landmarks_path);
    Scalar color(255, 0, 0);
    for( int i = 0; i < triangles.size(); i++){
        vector<Point2f> pt = triangles[i];
        line(img, pt[0], pt[1], color, 1, CV_AA, 0);
        line(img, pt[1], pt[2], color, 1, CV_AA, 0);
        line(img, pt[2], pt[0], color, 1, CV_AA, 0);
        putIndexInTriangle(img, pt, i);
    }
    imshow(landmarks_path, img);
    waitKey(0);
}

string get_average_landmarks_path(const string& image_path)
{
    string average_landmarks_path(image_path);
    string key = "/";
    size_t found = average_landmarks_path.rfind(key);
    average_landmarks_path.replace(found, average_landmarks_path.length() - found, "/average.csv");
    return average_landmarks_path;
}


// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );
    
    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> t1, vector<Point2f> t2)
{
    // Find bounding rectangle for each triangle
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);
    // Offset points by left top corner of the respective rectangles
    vector<Point2f> t1Rect, t2Rect;
    vector<Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {
        //tRect.push_back( Point2f( t[i].x - r.x, t[i].y -  r.y) );
        t2RectInt.push_back( Point((int)(t2[i].x - r2.x), (int)(t2[i].y - r2.y)) ); // for fillConvexPoly
        
        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
    }
    
    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    
    // Apply warpImage to small rectangular patches
    Mat img1Rect, img2Rect;
    img1(r1).copyTo(img1Rect);
    
    Mat warpImage = Mat::zeros(r2.height, r2.width, img1Rect.type());
    
    applyAffineTransform(warpImage, img1Rect, t1Rect, t2Rect);
    
    // Copy triangular region of the rectangular patch to the output image
    multiply(warpImage,mask, warpImage);
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + warpImage;
    
}

Mat averageFace(const vector<string>& paths, bool includeBoundaryPoints = true){
    Size size(200, 200);
    Mat averageImg(size, CV_32FC3);
    averageImg.convertTo(averageImg, CV_32FC3, 1/255.0);
    string average_landmarks_path = get_average_landmarks_path(paths[0]);
    
    vector<vector<int> > orderedTriangleListIndexes = getOrderedTriangleListIndexes(size, average_landmarks_path, includeBoundaryPoints);
    // unordered: trianglesToPointsVector(getDelaunayTriangles(size, average_landmarks_path), size); 
    vector<vector<Point2f> > averageDelaunayTriangles = getOrderedDelaunayTriangles(size, average_landmarks_path, orderedTriangleListIndexes, includeBoundaryPoints);
    int total_weights = 0;
    for (int i = 0; i < paths.size(); ++i)
    {
        string image_path = paths[i];
        string landmarks_path = image_path + ".csv";
        
        
        Mat img = imread(image_path);
        img.convertTo(img, CV_32FC3, 1/255.0);
        unordered: trianglesToPointsVector(getDelaunayTriangles(size, landmarks_path), size);
        vector<vector<Point2f> > triangles = getOrderedDelaunayTriangles(size, landmarks_path, orderedTriangleListIndexes, includeBoundaryPoints);

        Mat tempImg(size, CV_32FC3);
        for(int j = 0; j < triangles.size(); j++){
            vector<Point2f> triangle = triangles[j];
            vector<Point2f> averageTriangle = averageDelaunayTriangles[j];
            warpTriangle(img, tempImg, triangle, averageTriangle);
        }
        int weight = 1;
        total_weights += weight;
        averageImg += tempImg * weight;
    }
    averageImg /= total_weights;
    return averageImg;
}

Mat averageFace(const string path, bool includeBoundaryPoints = true){
    vector<string> paths;
    paths.push_back(path);
    return averageFace(paths, includeBoundaryPoints);
}

void darwNaiveAverageFace(const vector<string> paths){
    Mat averageImg;
    for(int i = 0; i < paths.size(); i++){
        string path = paths[i];
        Mat img = imread(path);
        img.convertTo(img, CV_32FC3, 1/255.0);
        if(i == 0){
            averageImg = img.clone();
        } else {
            averageImg += img;
        }
    }
    averageImg /= paths.size();
    imshow("Naieve Average", averageImg);
    waitKey(0);
}

Mat createMask(const Mat& img, const Size& size, const string& average_landmarks_path){
    vector<Point2f> points = getLandmarksPoints(size, average_landmarks_path, false);
    vector<int> hullIndex;
    vector<Point2f> hull;

    convexHull(points, hullIndex, false, false);
    for(int i = 0; i < hullIndex.size(); i++)
    {
        hull.push_back(points[hullIndex[i]]);
    }
    vector<Point> hull8U;
    for(int i = 0; i < hull.size(); i++)
    {
        Point pt(hull[i].x, hull[i].y);
        hull8U.push_back(pt);
    }
    Mat mask = Mat::zeros(size.width, size.height, img.depth());
    fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,255,255));

    return mask;    

}

void faceSwap(const string& path, Mat& averageImg){
    Mat img0Warpped = averageFace(path, true);
    averageImg.convertTo(averageImg, CV_8UC3, 255);
    img0Warpped.convertTo(img0Warpped, CV_8UC3, 255);
    boost::filesystem::create_directories("images");
    imwrite("images/averageImg.jpg", averageImg);
    imwrite("images/img0Warpped.jpg", img0Warpped);  
    Mat src = imread("images/averageImg.jpg");
    Mat dst = imread("images/img0Warpped.jpg");

    string average_landmarks_path = get_average_landmarks_path(path);
    Size size = averageImg.size();
    vector<Point2f> points = getLandmarksPoints(size, average_landmarks_path, false);
    vector<int> hullIndex;
    vector<Point2f> hull;

    convexHull(points, hullIndex, false, false);
    for(int i = 0; i < hullIndex.size(); i++)
    {
        hull.push_back(points[hullIndex[i]]);
    }
    vector<Point> hull8U;
    for(int i = 0; i < hull.size(); i++)
    {
        Point pt(hull[i].x, hull[i].y);
        hull8U.push_back(pt);
    }
    Mat mask = Mat::zeros(size.width, size.height, averageImg.depth());
    fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255,255,255));
    Rect r = boundingRect(hull);
    Point center = (r.tl() + r.br()) / 2;
    Mat output;
    seamlessClone(src, dst, mask, center, output, MIXED_CLONE);
    // imshow("Mask", mask);
    // averageImg.copyTo(img0Warpped, mask);
    // imshow("Copy to img", img0Warpped);
    imshow("img after warp", output);
}

int main(int argc, char** argv)
{  
    try
    {
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./delaunay_triangles_ex faces/*.jpg" << endl;
            return 0;
        }

        vector<string> paths;
        for(int i = 1; i < argc; i++){
            paths.push_back(string(argv[i]));
        }

        for(int i = 0; i < paths.size(); i++){
            String path = paths[i];
            Mat img = imread(path);
            // drawDelaunayTriangles(img, path + ".csv");  
            // drawLandmarks(path, path + ".csv");  
        }
        // darwNaiveAverageFace(paths);
        Mat averageImg = averageFace(paths, false);
        imshow("average", averageImg);
        faceSwap(paths[0], averageImg);
        waitKey(0);

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}