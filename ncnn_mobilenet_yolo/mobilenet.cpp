#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "net.h"
//#include "mat.h"
#include "platform.h"

#include "timer_counter.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_yolov3(const ncnn::Net& net, const cv::Mat& image, std::vector<Object>& objects)
{
    const float probThreshold {0.5};

    const int input_size = 352;

    int img_width = image.cols;
    int img_height = image.rows;

    ncnn::Mat ncnnImage = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_width, img_height, input_size, input_size);

    //subtract range mean value, norm to (-1, 1)
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    ncnnImage.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(4);

    ex.input("data", ncnnImage);

    ncnn::Mat detectRes;
    ex.extract("detection_out", detectRes);
    std::cout << detectRes.h << std::endl;

    objects.clear();

    for(int i = 0; i < detectRes.h; i++)
    {
        const float* values = detectRes.row(i);
        
        if(values[1] >= probThreshold)
        {
            Object object;
            object.label = values[0];
            object.prob = values[1];
            object.rect.x = values[2] * img_width;
            object.rect.y = values[3] * img_height;
            object.rect.width = values[4] * img_width - object.rect.x;
            object.rect.height = values[5] * img_height - object.rect.y;

            objects.push_back(object);
        }
        
    }

    return 0;

}

static void draw_objects(const cv::Mat& inputImage, const std::vector<Object>& objects)
{
    static const char* class_name[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};
    
    cv::Mat image = inputImage.clone();

    for(size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        cv::rectangle(image, obj.rect, cv::Scalar(50,205,50));

        std::string text {""};
        text += class_name[obj.label];
        text += std::to_string(obj.prob * 100);
        text += "%";

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, &baseline);

        int x = obj.rect.x;
        int y = obj.rect.y;
        if(y < 0)
            y = 0;
        if(x + textSize.width > image.cols)
            x = image.cols - textSize.width;
        
        //y + textSize.height for bottom left corner
        cv::rectangle(image, cv::Rect(x, y, textSize.width, textSize.height + 5), cv::Scalar(50,205,50), -1);
        cv::putText(image, text, cv::Point(x, y + textSize.height), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255,255,255));
        //cv::imshow("image", image);
        //cv::waitKey(0);
    }

    cv::imshow("image", image);
    //image only
    //cv::waitKey(0);
}

static int detect_from_image(const ncnn::Net& net, const std::string image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    //cv::imshow("image", image);
    //cv::waitKey(0);
    if (image.empty())
    {
        std::cerr << "no image has been read!" << std::endl;
        return -1;
    }

    TimerCounter tc;
    tc.Start();
    std::vector<Object> objects;
    //detection
    detect_yolov3(net, image, objects);
    tc.Stop();
    std::cout << "Time: " << tc.dbTime * 1000 << "ms" << std::endl;

    //draw box
    draw_objects(image, objects);

    return 0;
}

static int detect_from_video(const ncnn::Net& net, const std::string video_path)
{
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    cv::Mat frame;

    cv::VideoCapture cap(video_path);

    // Check if camera opened successfully
    if (!cap.isOpened()) 
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    while (true) 
    {
        // Capture frame-by-frame
        cap.read(frame);

        // If the frame is empty, break immediately
        if (frame.empty())
        {
            std::cout << "Error, blank frame.";
            break;
        }
        std::vector<Object> objects;

        // Display the resulting frame
        //imshow("Frame", frame);
        TimerCounter tc;
        tc.Start();
        detect_yolov3(net, frame, objects);
        tc.Stop();
        std::cout << "Time: " << tc.dbTime * 1000 << "ms" << std::endl;
        
        draw_objects(frame, objects);

        // Press ESC on keyboard to exit
        char c = (char)cv::waitKey(2);
        if (c == 27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}

static int detect_from_camera(const ncnn::Net& net)
{
    cv::Mat frame;

    cv::VideoCapture cap;

    // 0 = open default camera
    int deviceID = 0;
    // 0 = autodetect default API
    int apiID = cv::CAP_ANY;

    //open camera
    cap.open(deviceID + apiID);

    if(!cap.isOpened())
    {
        std::cerr<<"ERROR! Unable to open camera\n";
        return -1;
    }

    //--- grab and write loop
    while(true)
    {
        cap.read(frame);
        if(frame.empty())
        {
            std::cerr<<"ERROR! blank frame grabbed\n";
            return -1;
        }

        std::vector<Object> objects;
        TimerCounter tc;
        tc.Start();
        detect_yolov3(net, frame, objects);
        tc.Stop();
        std::cout << "Time: " << tc.dbTime * 1000 << "ms" << std::endl;
        
        draw_objects(frame, objects);
        
        //cv::imshow("image", frame);
        //char c = (char)cv::waitKey(20);
        //if (c == 27)
            //break;
        if(cv::waitKey(10) >= 0)
            break;
    }
    return 0;
}

int main(int argc, char** argv)
{
    std::string image_path = "./images/test3.jpg";
    std::string video_path = "./videos/test2.mp4";

    //load model and pretrained weights
    ncnn::Net net;
    net.load_param("./models/yolov3_m2-opt.param");
    net.load_model("./models/yolov3_m2-opt.bin");
    //net.load_param("./models/yolov3-opt.param");
    //net.load_model("./models/yolov3-opt.bin");

    //read image and recognize
    //detect_from_image(net, image_path);

    //detect_from_camera(net);

    detect_from_video(net, video_path);


    return 0;
}