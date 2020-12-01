// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//Kienct
#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>

// For viewing
#include <array>
#include <iostream>
#include <map>
#include <vector>
#include <BodyTrackingHelpers.h>
#include <Utilities.h>
#include <Window3dWrapper.h>

//Data Collecting and saving
#include <stdio.h>
#include <stdlib.h>
#include<fstream>
#include <iostream>
#include <windows.h>
#include <direct.h>
#define GetCurrentDir _getcwd
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    } 

std::string get_current_dir() {
    char buff[FILENAME_MAX]; //create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    string current_working_dir(buff);
    return current_working_dir;
}

void PrintUsage()
{
    printf("\nUSAGE: (k4abt_)simple_3d_viewer.exe SensorMode[NFOV_UNBINNED, WFOV_BINNED](optional) RuntimeMode[CPU](optional)\n");
    printf("  - SensorMode: \n");
    printf("      NFOV_UNBINNED (default) - Narrow Field of View Unbinned Mode [Resolution: 640x576; FOI: 75 degree x 65 degree]\n");
    printf("      WFOV_BINNED             - Wide Field of View Binned Mode [Resolution: 512x512; FOI: 120 degree x 120 degree]\n");
    printf("  - RuntimeMode: \n");
    printf("      CPU - Use the CPU only mode. It runs on machines without a GPU but it will be much slower\n");
    printf("      OFFLINE - Play a specified file. Does not require Kinect device\n");
    printf("e.g.   (k4abt_)simple_3d_viewer.exe WFOV_BINNED CPU\n");
    printf("e.g.   (k4abt_)simple_3d_viewer.exe CPU\n");
    printf("e.g.   (k4abt_)simple_3d_viewer.exe WFOV_BINNED\n");
    printf("e.g.   (k4abt_)simple_3d_viewer.exe OFFLINE MyFile.mkv\n");
}

void PrintAppUsage()
{
    printf("\n");
    printf(" Basic Navigation:\n\n");
    printf(" Rotate: Rotate the camera by moving the mouse while holding mouse left button\n");
    printf(" Pan: Translate the scene by holding Ctrl key and drag the scene with mouse left button\n");
    printf(" Zoom in/out: Move closer/farther away from the scene center by scrolling the mouse scroll wheel\n");
    printf(" Select Center: Center the scene based on a detected joint by right clicking the joint with mouse\n");
    printf("\n");
    printf(" Key Shortcuts\n\n");
    printf(" ESC: quit\n");
    printf(" h: help\n");
    printf(" b: body visualization mode\n");
    printf(" k: 3d window layout\n");
    printf("\n");
}

// Global State and Key Process Function
bool s_isRunning = true;
Visualization::Layout3d s_layoutMode = Visualization::Layout3d::OnlyMainView;
bool s_visualizeJointFrame = false;


int64_t ProcessKey(void* /*context*/, int key)
{
    // https://www.glfw.org/docs/latest/group__keys.html
    switch (key)
    {
        // Quit
    case GLFW_KEY_ESCAPE:
        s_isRunning = false;
        break;
    case GLFW_KEY_K:
        s_layoutMode = (Visualization::Layout3d)(((int)s_layoutMode + 1) % (int)Visualization::Layout3d::Count);
        break;
    case GLFW_KEY_B:
        s_visualizeJointFrame = !s_visualizeJointFrame;
        break;
    case GLFW_KEY_H:
        PrintAppUsage();
        break;
    }
    return 1;
}

int64_t CloseCallback(void* /*context*/)
{
    s_isRunning = false;
    return 1;
}

struct InputSettings
{
    k4a_depth_mode_t DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    bool CpuOnlyMode = false;
    bool Offline = false;
    std::string FileName;
};

bool ParseInputSettingsFromArg(int argc, char** argv, InputSettings& inputSettings)
{
    for (int i = 1; i < argc; i++)
    {
        std::string inputArg(argv[i]);
        if (inputArg == std::string("NFOV_UNBINNED"))
        {
            inputSettings.DepthCameraMode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        }
        else if (inputArg == std::string("WFOV_BINNED"))
        {
            inputSettings.DepthCameraMode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
        }
        else if (inputArg == std::string("CPU"))
        {
            inputSettings.CpuOnlyMode = true;
        }
        else if (inputArg == std::string("OFFLINE"))
        {
            inputSettings.Offline = true;
            if (i < argc - 1) {
                // Take the next argument after OFFLINE as file name
                inputSettings.FileName = argv[i + 1];
                i++;
            }
            else {
                return false;
            }
        }
        else
        {
            printf("Error command not understood: %s\n", inputArg.c_str());
            return false;
        }
    }
    return true;

}

void VisualizeResult(k4abt_frame_t bodyFrame, Window3dWrapper& window3d, int depthWidth, int depthHeight) {

    // Obtain original capture that generates the body tracking result
    k4a_capture_t originalCapture = k4abt_frame_get_capture(bodyFrame);
    k4a_image_t depthImage = k4a_capture_get_depth_image(originalCapture);

    std::vector<Color> pointCloudColors(depthWidth * depthHeight, { 1.f, 1.f, 1.f, 1.f });

    // Read body index map and assign colors
    k4a_image_t bodyIndexMap = k4abt_frame_get_body_index_map(bodyFrame);
    const uint8_t* bodyIndexMapBuffer = k4a_image_get_buffer(bodyIndexMap);
    for (int i = 0; i < depthWidth * depthHeight; i++)
    {
        uint8_t bodyIndex = bodyIndexMapBuffer[i];
        if (bodyIndex != K4ABT_BODY_INDEX_MAP_BACKGROUND)
        {
            uint32_t bodyId = k4abt_frame_get_body_id(bodyFrame, bodyIndex);
            pointCloudColors[i] = g_bodyColors[bodyId % g_bodyColors.size()];
        }
    }
    k4a_image_release(bodyIndexMap);

    // Visualize point cloud
    window3d.UpdatePointClouds(depthImage, pointCloudColors);

    // Visualize the skeleton data
    window3d.CleanJointsAndBones();
    uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
    for (uint32_t i = 0; i < numBodies; i++)
    {
        k4abt_body_t body;
        VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton), "Get skeleton from body frame failed!");
        body.id = k4abt_frame_get_body_id(bodyFrame, i);

        // Assign the correct color based on the body id
        Color color = g_bodyColors[body.id % g_bodyColors.size()];
        color.a = 0.4f;
        Color lowConfidenceColor = color;
        lowConfidenceColor.a = 0.1f;

        // Visualize joints
        for (int joint = 0; joint < static_cast<int>(K4ABT_JOINT_COUNT); joint++)
        {
            if (body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
            {
                const k4a_float3_t& jointPosition = body.skeleton.joints[joint].position;
                const k4a_quaternion_t& jointOrientation = body.skeleton.joints[joint].orientation;

                window3d.AddJoint(
                    jointPosition,
                    jointOrientation,
                    body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM ? color : lowConfidenceColor);
            }
        }

        // Visualize bones
        for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
        {
            k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
            k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

            if (body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW &&
                body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
            {
                bool confidentBone = body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM &&
                    body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM;
                const k4a_float3_t& joint1Position = body.skeleton.joints[joint1].position;
                const k4a_float3_t& joint2Position = body.skeleton.joints[joint2].position;

                window3d.AddBone(joint1Position, joint2Position, confidentBone ? color : lowConfidenceColor);
            }
        }
    }

    k4a_capture_release(originalCapture);
    k4a_image_release(depthImage);

}

void PlayFile(InputSettings inputSettings) {
    // Initialize the 3d window controller
    Window3dWrapper window3d;

    //create the tracker and playback handle
    k4a_calibration_t sensor_calibration;
    k4abt_tracker_t tracker = NULL;
    k4a_playback_t playback_handle = NULL;

    const char* file = inputSettings.FileName.c_str();
    if (k4a_playback_open(file, &playback_handle) != K4A_RESULT_SUCCEEDED)
    {
        printf("Failed to open recording: %s\n", file);
        return;
    }


    if (k4a_playback_get_calibration(playback_handle, &sensor_calibration) != K4A_RESULT_SUCCEEDED)
    {
        printf("Failed to get calibration\n");
        return;
    }
    

    k4a_capture_t capture = NULL;
    k4a_stream_result_t result = K4A_STREAM_RESULT_SUCCEEDED;

    k4abt_tracker_configuration_t tracker_config = { K4ABT_SENSOR_ORIENTATION_DEFAULT };

    tracker_config.processing_mode = inputSettings.CpuOnlyMode ? K4ABT_TRACKER_PROCESSING_MODE_CPU : K4ABT_TRACKER_PROCESSING_MODE_GPU;

    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker initialization failed!");

    k4abt_tracker_set_temporal_smoothing(tracker, 1);

    int depthWidth = sensor_calibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensor_calibration.depth_camera_calibration.resolution_height;

    window3d.Create("3D Visualization", sensor_calibration);
    window3d.SetCloseCallback(CloseCallback);
    window3d.SetKeyCallback(ProcessKey);

    while (result == K4A_STREAM_RESULT_SUCCEEDED)
    {
        result = k4a_playback_get_next_capture(playback_handle, &capture);
        // check to make sure we have a depth image
        // if we are not at the end of the file
        if (result != K4A_STREAM_RESULT_EOF) {
            k4a_image_t depth_image = k4a_capture_get_depth_image(capture);
            if (depth_image == NULL) {
                //If no depth image, print a warning and skip to next frame
                printf("Warning: No depth image, skipping frame\n");
                k4a_capture_release(capture);
                continue;
            }
            // Release the Depth image
            k4a_image_release(depth_image);
        }
        if (result == K4A_STREAM_RESULT_SUCCEEDED)
        {
            
            //enque capture and pop results - synchronous
            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, capture, K4A_WAIT_INFINITE);

            // Release the sensor capture once it is no longer needed.
            k4a_capture_release(capture);

            k4abt_frame_t bodyFrame = NULL;
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &bodyFrame, K4A_WAIT_INFINITE);
            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
            {
                size_t num_bodies = k4abt_frame_get_num_bodies(bodyFrame);
                printf("%zu bodies are detected\n", num_bodies);
                /************* Successfully get a body tracking result, process the result here ***************/
                VisualizeResult(bodyFrame, window3d, depthWidth, depthHeight); 
                //Release the bodyFrame
                k4abt_frame_release(bodyFrame);
            }
            else
            {
                printf("Pop body frame result failed!\n");
                break;
            }
           
        }

        window3d.SetLayout3d(s_layoutMode);
        window3d.SetJointFrameVisualization(s_visualizeJointFrame);
        window3d.Render();

        if (result == K4A_STREAM_RESULT_EOF)
        {
            // End of file reached
            break;
        }
    }
    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);
    window3d.Delete();
    printf("Finished body tracking processing!\n");
    k4a_playback_close(playback_handle);

}

void PlayFromDevice(InputSettings inputSettings) {

    fstream file; //object of fstream class

   //opening file "sample.txt" in out(write) mode
    file.open("../../../data/skeletons/sample.txt", ios::out);
    if (!file)
    {
        cout << get_current_dir();
        cout << "Error in creating file!!!" << endl;
    }
    else {
        cout << "File created successfully." << endl;
    }


    k4a_device_t device = nullptr;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    //deviceConfig.depth_mode = inputSettings.DepthCameraMode;
    //deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_30;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32; // <==== For Color image
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_2160P;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED; // <==== For Depth image 
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    // Get calibration information
    k4a_calibration_t sensorCalibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensorCalibration),
        "Get depth camera calibration failed!");
    int depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;

    // Create Body Tracker
    k4abt_tracker_t tracker = nullptr;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    tracker_config.processing_mode = inputSettings.CpuOnlyMode ? K4ABT_TRACKER_PROCESSING_MODE_CPU : K4ABT_TRACKER_PROCESSING_MODE_GPU;
    VERIFY(k4abt_tracker_create(&sensorCalibration, tracker_config, &tracker), "Body tracker initialization failed!");
    // Initialize the 3d window controller
    Window3dWrapper window3d;
    window3d.Create("3D Visualization", sensorCalibration);
    window3d.SetCloseCallback(CloseCallback);
    window3d.SetKeyCallback(ProcessKey);

    bool exit = false;
    int frame_count = 0;

    while (exit == false)
    {
        k4a_capture_t sensorCapture = nullptr;
        k4a_wait_result_t getCaptureResult = k4a_device_get_capture(device, &sensorCapture, K4A_WAIT_INFINITE); 

        if (getCaptureResult == K4A_WAIT_RESULT_SUCCEEDED)
        {   // Record RGB Image
            k4a_image_t colorImage = k4a_capture_get_color_image(sensorCapture);
            if (colorImage != NULL)
            {
                // you can check the format with this function
                k4a_image_format_t format = k4a_image_get_format(colorImage); // K4A_IMAGE_FORMAT_COLOR_BGRA32 

                // get raw buffer
                uint8_t* buffer = k4a_image_get_buffer(colorImage);

                // convert the raw buffer to cv::Mat
                int rows = k4a_image_get_height_pixels(colorImage);
                int cols = k4a_image_get_width_pixels(colorImage);
                cv::Mat colorMat(rows, cols, CV_8UC4, (void*)buffer, cv::Mat::AUTO_STEP);

                // Release the image
                k4a_image_release(colorImage);

                //save RGB image
                string name = "../../../data/images/img" + to_string(frame_count) + ".png";
                imwrite(name, colorMat);
            }

            k4a_image_t depthImage = k4a_capture_get_depth_image(sensorCapture); // get image metadata
            if (depthImage != NULL)
            {
                // you can check the format with this function
                k4a_image_format_t format = k4a_image_get_format(depthImage); // K4A_IMAGE_FORMAT_DEPTH16 

                // get raw buffer
                uint8_t* buffer = k4a_image_get_buffer(depthImage);

                // convert the raw buffer to cv::Mat
                int rows = k4a_image_get_height_pixels(depthImage);
                int cols = k4a_image_get_width_pixels(depthImage);
                cv::Mat depthMat(rows, cols, CV_16U, (void*)buffer, cv::Mat::AUTO_STEP);
                //cv::Mat depthMat = cv::Mat(rows, cols, CV_16UC1, reinterpret_cast<uint16_t*>(buffer));
                //cv::Mat depthMat(rows, cols, CV_8UC1, (void*)buffer, cv::Mat::AUTO_STEP);

                k4a_image_release(depthImage);
                Mat gray = depthMat;

                string name = "../../../data/depth/depth" + to_string(frame_count) + ".png";
                imwrite(name, gray);
                
            }
            
            frame_count++;

            // timeout_in_ms is set to 0. Return immediately no matter whether the sensorCapture is successfully added
            // to the queue or not.
            k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(tracker, sensorCapture, 0);

            // Release the sensor capture once it is no longer needed.
            k4a_capture_release(sensorCapture);

            if (queueCaptureResult == K4A_WAIT_RESULT_FAILED)
            {
                std::cout << "Error! Add capture to tracker process queue failed!" << std::endl;
                break;
            }
        }
        else if (getCaptureResult != K4A_WAIT_RESULT_TIMEOUT)
        {
            std::cout << "Get depth capture returned error: " << getCaptureResult << std::endl;
            break;
        }

        // Pop Result from Body Tracker
        k4abt_frame_t bodyFrame = nullptr;
        k4a_wait_result_t popFrameResult = k4abt_tracker_pop_result(tracker, &bodyFrame, K4A_WAIT_INFINITE); // timeout_in_ms is set to 0
        if (popFrameResult == K4A_WAIT_RESULT_SUCCEEDED)
        {
            /************* Successfully get a body tracking result, process the result here ***************/
            VisualizeResult(bodyFrame, window3d, depthWidth, depthHeight);

            // Record Skeleton data
            size_t num_bodies = k4abt_frame_get_num_bodies(bodyFrame);
            printf("%zu bodies are detected!\n", num_bodies);

            for (int i = 0; i < num_bodies; i++)
            {
                k4abt_skeleton_t skeleton;

                k4a_result_t bodyResult = k4abt_frame_get_body_skeleton(bodyFrame, i, &skeleton);

                if (bodyResult == K4A_RESULT_SUCCEEDED)
                {
                    int jointCount = static_cast<int> (K4ABT_JOINT_COUNT);
                    for (int joint = 0; joint < jointCount; joint++) {
                        const k4a_float3_t& position = skeleton.joints[joint].position;
                        const k4a_quaternion_t& orient = skeleton.joints[joint].orientation;
                        file << position.xyz.x << "," << position.xyz.y << "," << position.xyz.z << ",";
                        file << orient.wxyz.x << "," << orient.wxyz.y << "," << orient.wxyz.z << ",";
                        file << skeleton.joints[joint].confidence_level << "\n";
                    }
                    file << ";" << "\n";
                    //k4abt_joint_t head = skeleton.joints[K4ABT_JOINT_NOSE];
                    //k4a_float3_t position = head.position;
                    //k4a_quaternion_t ori = head.orientation;
                    //k4abt_joint_confidence_level_t con = head.confidence_level;
                    //printf("%l32u is the confidence\n",jointCount);
                   // printf("%.7lf is the Orientation\n", ori.wxyz.w);
                   // printf("%.7lf is the Position\n", position.xyz.x);
                   // file << position.xyz.x << ",";
                   // file << position.xyz.y << "\n";
                }
            }

            //Release the bodyFrame
            k4abt_frame_release(bodyFrame);
        }
       
        window3d.SetLayout3d(s_layoutMode);
        window3d.SetJointFrameVisualization(s_visualizeJointFrame);
        window3d.Render();

        if (GetAsyncKeyState(VK_ESCAPE))
        {
            exit = true;
        }
    }

    std::cout << "Finished body tracking processing!" << std::endl;

    window3d.Delete();
    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);


}

int main(int argc, char** argv)
{
    InputSettings inputSettings;
   
    if (ParseInputSettingsFromArg(argc, argv, inputSettings)) {
        // Either play the offline file or play from the device
        if (inputSettings.Offline == true) {     
            PlayFile(inputSettings);
        }
        else {
            PlayFromDevice(inputSettings);
        }
    }
    else {
        // Print app usage if user entered incorrect arguments.
        PrintUsage();
        return -1;
    }

    return 0;
}
