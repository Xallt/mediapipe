// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#include "run_graph_gpu.h"

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/util/resource_util.h"

// include for cv::Mat and cv::Capture
#include <opencv2/opencv.hpp>

#define NormalizedLandmarkList ::mediapipe::NormalizedLandmarkList

constexpr char kInputStream[] = "input_video";
constexpr char kWindowName[] = "MediaPipe";

absl::Status CreateGraphFromFile(std::string calculator_graph_config_file, mediapipe::CalculatorGraph &graph) {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        calculator_graph_config_file,
        &calculator_graph_config_contents));
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    MP_RETURN_IF_ERROR(graph.Initialize(config));

    return absl::OkStatus();
}

class SimpleVideoReader {
   public:
    SimpleVideoReader();
    ~SimpleVideoReader();
    absl::Status init(std::string url, bool flipFrame = false);
    absl::Status getFrame(cv::Mat &frame);
    double getFPS();
    void setFPS(double fps);
    void setResolution(int width, int height);

   private:
    cv::VideoCapture capture;
    bool flipFrame;
};
// SimpleVideoWrapper class for video input from webcam / video file
SimpleVideoReader::SimpleVideoReader() {}

SimpleVideoReader::~SimpleVideoReader() {
    if (capture.isOpened()) capture.release();
}

absl::Status SimpleVideoReader::init(std::string url, bool flipFrame) {
    if (url == "0")
        capture.open(0);
    else
        capture.open(url);

    if (!capture.isOpened()) {
        return absl::InternalError(
            absl::StrCat("Failed to open video capture for URL: ", url));
    }

    this->flipFrame = flipFrame;
    return absl::OkStatus();
}

absl::Status SimpleVideoReader::getFrame(cv::Mat &frame) {
    bool ret = capture.read(frame);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);
    if (!ret) {
        return absl::InternalError("Failed to read frame from capture.");
    }

    if (flipFrame) cv::flip(frame, frame, 1);
    return absl::OkStatus();
}

double SimpleVideoReader::getFPS() {
    return capture.get(cv::CAP_PROP_FPS);
}

void SimpleVideoReader::setFPS(double fps) {
    capture.set(cv::CAP_PROP_FPS, fps);
}

void SimpleVideoReader::setResolution(int width, int height) {
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
}

const char kVideoOutputStream[] = "output_video";
const char kLandmarksOutputStream[] = "hand_landmarks";
const char kLandmarkPresenceOutputStream[] = "landmark_presence";
class MPPGraphRunner {
   public:
    absl::Status InitMPPGraph(std::string calculator_graph_config_file) {
        MP_RETURN_IF_ERROR(CreateGraphFromFile(calculator_graph_config_file, graph));

        MP_ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
        MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));

        gpu_helper.InitializeForTest(graph.GetGpuResources().get());

        MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_video_tmp,
                            graph.AddOutputStreamPoller(kVideoOutputStream));
        poller_video = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_video_tmp));

        MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_tmp,
                            graph.AddOutputStreamPoller(kLandmarksOutputStream));
        poller_landmarks = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmarks_tmp));

        MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark_presence_tmp,
                            graph.AddOutputStreamPoller(kLandmarkPresenceOutputStream));
        poller_landmark_presence = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmark_presence_tmp));
        MP_RETURN_IF_ERROR(graph.StartRun({}));
        return absl::OkStatus();
    }
    absl::Status ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, cv::Mat &output_frame_mat, std::vector<NormalizedLandmarkList> &landmarks, bool &landmark_presence) {
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);
        MP_RETURN_IF_ERROR(
            gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, this]() -> absl::Status {
                // Convert ImageFrame to GpuBuffer.
                auto texture = this->gpu_helper.CreateSourceTexture(*input_frame.get());
                auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
                glFlush();
                texture.Release();
                // Send GPU image packet into the graph.
                // MP_RETURN_IF_ERROR(
                auto status = this->graph.AddPacketToInputStream(
                    kInputStream, mediapipe::Adopt(gpu_frame.release())
                                      .At(mediapipe::Timestamp(frame_timestamp_us)));
                ABSL_LOG(INFO) << status;
                // );
                return absl::OkStatus();
            }));

        // Get the graph result packet, or stop if that fails.
        mediapipe::Packet packet_video, packet_landmarks, packet_landmark_presence;
        poller_video->Next(&packet_video);

        poller_landmark_presence->Next(&packet_landmark_presence);
        landmark_presence = packet_landmark_presence.Get<bool>();
        if (landmark_presence) {
            poller_landmarks->Next(&packet_landmarks);
            landmarks = packet_landmarks.Get<std::vector<NormalizedLandmarkList>>();
        }

        std::unique_ptr<mediapipe::ImageFrame> output_frame;

        // landmarks = packet_landmarks.Get<std::vector<NormalizedLandmarkList>>();

        // Convert GpuBuffer to ImageFrame.
        MP_RETURN_IF_ERROR(
            gpu_helper.RunInGlContext(
                [&packet_video, &output_frame, this]() -> absl::Status {
                    auto &gpu_frame = packet_video.Get<mediapipe::GpuBuffer>();
                    auto texture = this->gpu_helper.CreateSourceTexture(gpu_frame);
                    output_frame = absl::make_unique<mediapipe::ImageFrame>(
                        mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
                        gpu_frame.width(), gpu_frame.height(),
                        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
                    this->gpu_helper.BindFramebuffer(texture);
                    const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
                        gpu_frame.format(), 0, gpu_helper.GetGlVersion());
                    glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                                 info.gl_type, output_frame->MutablePixelData());
                    glFlush();
                    texture.Release();
                    return absl::OkStatus();
                }));

        // Convert back to opencv for display or saving.
        output_frame_mat = mediapipe::formats::MatView(output_frame.get());
        if (output_frame_mat.channels() == 4)
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
        else
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        return absl::OkStatus();
    }
    absl::Status RunMPPGraph(std::string calculator_graph_config_file, std::string input_video_path, std::string output_video_path) {
        ABSL_LOG(INFO) << "Initialize&Start the calculator graph.";
        MP_RETURN_IF_ERROR(InitMPPGraph(calculator_graph_config_file));

        ABSL_LOG(INFO) << "Initialize the camera or load the video.";
        if (input_video_path.empty())
            input_video_path = "0";
        SimpleVideoReader capture;
        MP_RETURN_IF_ERROR(capture.init(input_video_path, input_video_path == "0"));

        cv::VideoWriter writer;
        const bool save_video = !output_video_path.empty();
        if (!save_video) {
            cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
            capture.setResolution(640, 480);
            capture.setFPS(30);
#endif
        }

        ABSL_LOG(INFO) << "Start grabbing and processing frames.";
        bool grab_frames = true;
        while (grab_frames) {
            // Capture opencv camera or video frame.
            cv::Mat camera_frame;
            MP_RETURN_IF_ERROR(capture.getFrame(camera_frame));

            // Prepare and add graph input packet.
            size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

            cv::Mat output_frame_mat;
            std::vector<NormalizedLandmarkList> landmarks;
            bool landmark_presence;
            MP_RETURN_IF_ERROR(ProcessFrame(camera_frame, frame_timestamp_us, output_frame_mat, landmarks, landmark_presence));

            // Wrap Mat into an ImageFrame.
            if (save_video) {
                if (!writer.isOpened()) {
                    ABSL_LOG(INFO) << "Prepare video writer.";
                    writer.open(output_video_path,
                                mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                                capture.getFPS(), output_frame_mat.size());
                    RET_CHECK(writer.isOpened());
                }
                writer.write(output_frame_mat);
            } else {
                cv::imshow(kWindowName, output_frame_mat);
                // Press any key to exit.
                const int pressed_key = cv::waitKey(5);
                if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
            }
        }

        ABSL_LOG(INFO) << "Shutting down.";
        if (writer.isOpened()) writer.release();
        MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
        return graph.WaitUntilDone();
    }

   private:
    mediapipe::CalculatorGraph graph;
    mediapipe::GlCalculatorHelper gpu_helper;
    std::unique_ptr<mediapipe::OutputStreamPoller> poller_video;
    std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmarks;
    std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmark_presence;
};

SimpleMPPGraphRunner::SimpleMPPGraphRunner() {}
bool SimpleMPPGraphRunner::RunMPPGraph(std::string calculator_graph_config_file, std::string input_video_path, std::string output_video_path) {
    runnerVoid = (void *)new MPPGraphRunner();
    MPPGraphRunner &runner = *(MPPGraphRunner *)runnerVoid;
    absl::Status status = runner.RunMPPGraph(calculator_graph_config_file, input_video_path, output_video_path);
    if (!status.ok())
        std::cout << "Failed to run the graph: " << status.message() << std::endl;

    return status.ok();
}
bool SimpleMPPGraphRunner::InitMPPGraph(std::string calculator_graph_config_file) {
    runnerVoid = (void *)new MPPGraphRunner();
    MPPGraphRunner &runner = *(MPPGraphRunner *)runnerVoid;
    absl::Status status = runner.InitMPPGraph(calculator_graph_config_file);
    if (!status.ok())
        std::cout << "Failed to initialize the graph: " << status.message() << std::endl;

    return status.ok();
}
bool SimpleMPPGraphRunner::ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, cv::Mat &output_frame_mat, std::vector<LandmarkList> &landmarks, bool &landmark_presence) {
    MPPGraphRunner &runner = *(MPPGraphRunner *)runnerVoid;
    std::vector<NormalizedLandmarkList> landmarks_tmp;
    absl::Status status = runner.ProcessFrame(camera_frame, frame_timestamp_us, output_frame_mat, landmarks_tmp, landmark_presence);
    if (!status.ok()) {
        std::cout << "Failed to process the frame: " << status.message() << std::endl;
        return false;
    }

    landmarks.resize(landmarks_tmp.size());
    for (int i = 0; i < landmarks_tmp.size(); i++) {
        landmarks[i].landmarks.resize(landmarks_tmp[i].landmark_size());
        landmarks[i].presence.resize(landmarks_tmp[i].landmark_size());
        landmarks[i].visibility.resize(landmarks_tmp[i].landmark_size());
        for (int j = 0; j < landmarks_tmp[i].landmark_size(); j++) {
            landmarks[i].landmarks[j].x = landmarks_tmp[i].landmark(j).x();
            landmarks[i].landmarks[j].y = landmarks_tmp[i].landmark(j).y();
            landmarks[i].landmarks[j].z = landmarks_tmp[i].landmark(j).z();
            landmarks[i].presence[j] = landmarks_tmp[i].landmark(j).presence();
            landmarks[i].visibility[j] = landmarks_tmp[i].landmark(j).visibility();
        }
    }

	return status.ok();
}
SimpleMPPGraphRunner::~SimpleMPPGraphRunner() {
	delete (MPPGraphRunner *)runnerVoid;
}