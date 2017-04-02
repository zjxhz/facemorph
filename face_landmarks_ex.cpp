#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>

using namespace dlib;
using namespace std;


string get_face_path(const string& image_path)
{
    string face_path(image_path);
    string key = "/";
    size_t found = face_path.rfind(key);
    face_path.replace(found, key.length(), "/cropped/");
    found = face_path.rfind(key);
    string folder = face_path.substr(0, found);
    boost::filesystem::create_directories(folder);
    return face_path;
};

std::vector<full_object_detection> get_landmarks(const string& predictor, const array2d<rgb_pixel>& img)
{
    shape_predictor sp;
    deserialize(predictor) >> sp;
    frontal_face_detector detector = get_frontal_face_detector();   
    // pyramid_up(img);// Make the image larger so we can detect small faces. TODO: is this needed?
    std::vector<dlib::rectangle> dets = detector(img);//all faces
    std::vector<full_object_detection> shapes;
    for (unsigned long j = 0; j < dets.size(); ++j)
    {
        full_object_detection shape = sp(img, dets[j]);
        shapes.push_back(shape);
    }
    return shapes;
}

std::vector<full_object_detection> get_landmarks(const shape_predictor& sp, const array2d<rgb_pixel>& img)
{
    // pyramid_up(img);// Make the image larger so we can detect small faces. TODO: is this needed?
    frontal_face_detector detector = get_frontal_face_detector();  
    std::vector<dlib::rectangle> dets = detector(img);//all faces
    std::vector<full_object_detection> shapes;
    for (unsigned long j = 0; j < dets.size(); ++j)
    {
        full_object_detection shape = sp(img, dets[j]);
        shapes.push_back(shape);
    }
    return shapes;
}

// cropp, rotat upright, and scaled to a standard size
void extract_faces(const array2d<rgb_pixel>& img, std::vector<full_object_detection> shapes, dlib::array<array2d<rgb_pixel> >& faces)
{
    extract_image_chips(img, get_face_chip_details(shapes), faces);
}


string get_landmarks_path(const string& image_path)
{
    string landmarks_path(image_path + ".csv");
    return get_face_path(landmarks_path);
}

string get_average_landmarks_path(const string& image_path)
{
	string landmarks_path = get_landmarks_path(image_path);
	string key = "/";
	size_t found = landmarks_path.rfind(key);
	landmarks_path.replace(found, landmarks_path.length() - found, "/average.csv");
	return landmarks_path;
}

void points_to_file(const std::vector<point> points, const string& file)
{
	ofstream out;
	out.open (file);

    for(int j =0; j < points.size(); j++){
    	point p = points[j];
        out << p.x()
            << " "
            << p.y()
        	<< endl;
    }
    
    out.close();
}

void landmakrs_to_file(const std::vector<full_object_detection>& all_landmarks, const string& file)
{
	std::vector<point> points;
	
	for(int i = 0; i < all_landmarks[0].num_parts(); i++){
		int averageX = 0;
		int averageY = 0;
		int size = all_landmarks.size();
		int total_weights = 0;
		for(int j = 0; j < size; j++){
			int weight = 1;//pow((j + 1), 2);
			averageX += all_landmarks[j].part(i).x() * weight;
			averageY += all_landmarks[j].part(i).y() * weight;
			total_weights += weight;
		}
		points.push_back(point(averageX / total_weights, averageY / total_weights));
	}

    points_to_file(points, file);
}

void landmakrs_to_file(const full_object_detection& landmarks, const string& file)
{
	std::vector<point> points;
	
    for(int i =0; i < landmarks.num_parts(); i++){
    	points.push_back(point(landmarks.part(i).x(), landmarks.part(i).y()));
    }
    
    points_to_file(points, file);
}

int main(int argc, char** argv)
{  
    try
    {
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }
        string predictor = argv[1];
        std::vector<full_object_detection> all_landmarks;
        shape_predictor sp;
		deserialize(predictor) >> sp;
		frontal_face_detector detector = get_frontal_face_detector();   
        for (int i = 2; i < argc; ++i)
        {
            dlib::array2d<rgb_pixel> img;
            string image_path = argv[i];
            cout << "Processing " << image_path << endl;
            load_image(img, image_path);           
            std::vector<full_object_detection> landmarks = get_landmarks(sp, img);

            dlib::array<array2d<rgb_pixel> > faces;
            extract_faces(img, landmarks, faces);
            
            string cropped_face_path = get_face_path(argv[i]);

            save_jpeg(tile_images(faces), cropped_face_path);
            array2d<rgb_pixel> saved_img;
            load_image(saved_img, cropped_face_path);

            landmarks = get_landmarks(sp, saved_img);//assume there is only one face detected 
            all_landmarks.push_back(landmarks[0]);

            landmakrs_to_file(landmarks[0], get_landmarks_path(argv[i]));
        }
        landmakrs_to_file(all_landmarks, get_average_landmarks_path(argv[2]));
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}
