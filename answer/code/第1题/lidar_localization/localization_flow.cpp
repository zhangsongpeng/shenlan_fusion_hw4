#include "lidar_localization/matching/localization_flow.hpp"

#include "glog/logging.h"
#include "lidar_localization/global_defination/global_defination.h"
#include <random>
#include "lidar_localization/tools/file_manager.hpp"
namespace lidar_localization {
LocalizationFlow::LocalizationFlow(ros::NodeHandle& nh) {
    // subscriber
    cloud_sub_ptr_ = std::make_shared<CloudSubscriber>(nh, "/kitti/velo/pointcloud", 100000);
    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, "/kitti/oxts/imu", 1000000);
    velocity_sub_ptr_ = std::make_shared<VelocitySubscriber>(nh, "/kitti/oxts/gps/vel", 1000000);
    gnss_sub_ptr_ = std::make_shared<GNSSSubscriber>(nh, "/kitti/oxts/gps/fix", 1000000);
    lidar_to_imu_ptr_ = std::make_shared<TFListener>(nh, "/imu_link", "/velo_link");

    distortion_adjust_ptr_ = std::make_shared<DistortionAdjust>();
    // publisher
    global_map_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "/global_map", "/map", 100);
    local_map_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "/local_map", "/map", 100);
    current_scan_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "/current_scan", "/map", 100);
    laser_odom_pub_ptr_ = std::make_shared<OdometryPublisher>(nh, "/laser_localization", "/map", "/lidar", 100);
    laser_tf_pub_ptr_ = std::make_shared<TFBroadCaster>("/map", "/vehicle_link");
    gnss_pub_ptr_ = std::make_shared<OdometryPublisher>(nh, "/synced_gnss", "/map", "/imu_link", 100);

    matching_ptr_ = std::make_shared<Matching>();
    std::string config_file_path = WORK_SPACE_PATH + "/config/matching/kalman_filter.yaml";
    YAML::Node config_node = YAML::LoadFile(config_file_path);
    gyro_noise_ = config_node["gyro_noise"].as<double>();
    acc_noise_ = config_node["acc_noise"].as<double>();
    dp_noise_ = config_node["dp_noise"].as<double>();
    dphi_noise_ = config_node["dphi_noise"].as<double>();

    FileManager::CreateDirectory(WORK_SPACE_PATH + "/localization_data");
    FileManager::CreateFile(ground_truth_ofs_, WORK_SPACE_PATH + "/localization_data/ground_truth.txt");
    FileManager::CreateFile(localization_ofs_, WORK_SPACE_PATH + "/localization_data/localization.txt");
}

bool LocalizationFlow::Run() {
    if (matching_ptr_->HasNewGlobalMap() && global_map_pub_ptr_->HasSubscribers()) {
        CloudData::CLOUD_PTR global_map_ptr(new CloudData::CLOUD());
        matching_ptr_->GetGlobalMap(global_map_ptr);
        global_map_pub_ptr_->Publish(global_map_ptr);
    }

    if (matching_ptr_->HasNewLocalMap() && local_map_pub_ptr_->HasSubscribers())
        local_map_pub_ptr_->Publish(matching_ptr_->GetLocalMap());

    if (!ReadData())
        return false;

    if (!InitCalibration()) 
        return false;

    if (!InitGNSS())
        return false;

    if (!InitPose())
        return false;

    while(SyncData(true)) {
        Filter();
        TransformData();
        PublishData();
    }

    return true;
}

bool LocalizationFlow::ReadData() {
    static bool sensor_inited = false;
    cloud_sub_ptr_->ParseData(cloud_data_buff_);
    imu_sub_ptr_->ParseData(imu_data_buff_);
    velocity_sub_ptr_->ParseData(velocity_data_buff_);
    gnss_sub_ptr_->ParseData(gnss_data_buff_);

    if (cloud_data_buff_.empty() || imu_data_buff_.empty()
    || velocity_data_buff_.empty() || gnss_data_buff_.empty())
        return false;

    if (!sensor_inited)
    {
        while (!cloud_data_buff_.empty())
        {
            if (imu_data_buff_.front().time > cloud_data_buff_.front().time
            || velocity_data_buff_.front().time > cloud_data_buff_.front().time
            || gnss_data_buff_.front().time > cloud_data_buff_.front().time)
            {
                cloud_data_buff_.pop_front();
            }
            else
            {
                sensor_inited = true;
                break;
            }
        }
    }

    return sensor_inited;
}

bool LocalizationFlow::InitCalibration() {
    static bool calibration_received = false;
    if (!calibration_received) {
        if (lidar_to_imu_ptr_->LookupData(lidar_to_imu_)) {
            calibration_received = true;
        }
    }

    return calibration_received;
}

bool LocalizationFlow::InitGNSS() {
    static bool gnss_inited = false;
    if (!gnss_inited) {
        // GNSSData gnss_data = gnss_data_buff_.front();
        GNSSData gnss_data;
        gnss_data.latitude = 48.9826576154;
        gnss_data.longitude = 8.39045533533;
        gnss_data.altitude = 116.39641207;
        // latitude = 48.9852365396;
        // longitude = 8.39364241238;
        // altitude = 116.372802178;
        gnss_data.InitOriginPosition();
        gnss_inited = true;
    }

    return gnss_inited;
}

bool LocalizationFlow::InitPose() {
    static bool pose_inited = false;
    if (pose_inited)
    {
        return true;
    }
    if (!SyncData(false))
    {
        return false;
    }
    TransformData();
    Eigen::Matrix4f laser_odometry;
    laser_odometry = gnss_pose_ * lidar_to_imu_;
    matching_ptr_->SetGNSSPose(laser_odometry);
    matching_ptr_->Update(current_cloud_data_, laser_odometry);
    matching_ptr_->TransformCurrentScan(current_cloud_data_, laser_odometry);
    laser_odometry = laser_odometry * lidar_to_imu_.inverse();
    state_.p = laser_odometry.block<3, 1>(0, 3).cast<double>();
    state_.q = Eigen::Quaterniond(laser_odometry.block<3, 3>(0, 0).cast<double>());
    state_.v[0] = current_velocity_data_.linear_velocity.x;
    state_.v[1] = current_velocity_data_.linear_velocity.y;
    state_.v[2] = current_velocity_data_.linear_velocity.z;
    state_.v = state_.q * state_.v;
    state_.bg = Eigen::Vector3d(0, 0, 0);
    state_.ba = Eigen::Vector3d(0, 0, 0);
    error_state_.x.setZero();
    error_state_.p.setZero();
    error_state_.p.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-2;
    error_state_.p.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-3;
    error_state_.p.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 3e-4;
    error_state_.p.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-4;
    error_state_.p.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-6;
    pose_inited = true;
    // imu_data_buff_.pop_front();
    // cloud_data_buff_.pop_front();
    TransformData();
    PublishData();
    return true;
}

bool LocalizationFlow::SyncData(bool inited)
{
    if (cloud_data_buff_.empty())
    {
        return false;
    }
    current_cloud_data_ = cloud_data_buff_.front();
    double sync_time = current_cloud_data_.time;
    while (gnss_data_buff_.size() > 1)
    {
        if (gnss_data_buff_[1].time < sync_time)
        {
            gnss_data_buff_.pop_front();
        }
        else
        {
            break;
        }
    }

    if (gnss_data_buff_.size() > 1)
    {
        GNSSData front_data = gnss_data_buff_.at(0);
        GNSSData back_data = gnss_data_buff_.at(1);
        double front_scale = (back_data.time - sync_time) / (back_data.time - front_data.time);
        double back_scale = (sync_time - front_data.time) / (back_data.time - front_data.time);
        current_gnss_data_.time = sync_time;
        current_gnss_data_.status = back_data.status;
        current_gnss_data_.longitude = front_data.longitude * front_scale + back_data.longitude * back_scale;
        current_gnss_data_.latitude = front_data.latitude * front_scale + back_data.latitude * back_scale;
        current_gnss_data_.altitude = front_data.altitude * front_scale + back_data.altitude * back_scale;
        current_gnss_data_.local_E = front_data.local_E * front_scale + back_data.local_E * back_scale;
        current_gnss_data_.local_N = front_data.local_N * front_scale + back_data.local_N * back_scale;
        current_gnss_data_.local_U = front_data.local_U * front_scale + back_data.local_U * back_scale;
    }
    else
    {
        return false;
    }

    while (velocity_data_buff_.size() > 1)
    {
        if (velocity_data_buff_[1].time < sync_time)
        {
            velocity_data_buff_.pop_front();
        }
        else
        {
            break;
        }
    }

    if (velocity_data_buff_.size() > 1)
    {
        VelocityData front_data = velocity_data_buff_.at(0);
        VelocityData back_data = velocity_data_buff_.at(1);

        double front_scale = (back_data.time - sync_time) / (back_data.time - front_data.time);
        double back_scale = (sync_time - front_data.time) / (back_data.time - front_data.time);
        current_velocity_data_.time = sync_time;
        current_velocity_data_.linear_velocity.x = front_data.linear_velocity.x * front_scale + back_data.linear_velocity.x * back_scale;
        current_velocity_data_.linear_velocity.y = front_data.linear_velocity.y * front_scale + back_data.linear_velocity.y * back_scale;
        current_velocity_data_.linear_velocity.z = front_data.linear_velocity.z * front_scale + back_data.linear_velocity.z * back_scale;
        current_velocity_data_.angular_velocity.x = front_data.angular_velocity.x * front_scale + back_data.angular_velocity.x * back_scale;
        current_velocity_data_.angular_velocity.y = front_data.angular_velocity.y * front_scale + back_data.angular_velocity.y * back_scale;
        current_velocity_data_.angular_velocity.z = front_data.angular_velocity.z * front_scale + back_data.angular_velocity.z * back_scale;
    }
    else
    {
        return false;
    }

    while (!inited && imu_data_buff_.size() > 1)
    {
        if (imu_data_buff_[1].time < sync_time)
        {
            imu_data_buff_.pop_front();
        }
        else
        {
            break;
        }
    }
    
    if (imu_data_buff_.size() > 1)
    {
        if (!inited)
        {
            current_imu_data_.clear();
            IMUData front_data = imu_data_buff_.at(0);
            IMUData back_data = imu_data_buff_.at(1);
            IMUData synced_data;

            double front_scale = (back_data.time - sync_time) / (back_data.time - front_data.time);
            double back_scale = (sync_time - front_data.time) / (back_data.time - front_data.time);
            synced_data.time = sync_time;
            synced_data.linear_acceleration.x = front_data.linear_acceleration.x * front_scale + back_data.linear_acceleration.x * back_scale;
            synced_data.linear_acceleration.y = front_data.linear_acceleration.y * front_scale + back_data.linear_acceleration.y * back_scale;
            synced_data.linear_acceleration.z = front_data.linear_acceleration.z * front_scale + back_data.linear_acceleration.z * back_scale;
            synced_data.angular_velocity.x = front_data.angular_velocity.x * front_scale + back_data.angular_velocity.x * back_scale;
            synced_data.angular_velocity.y = front_data.angular_velocity.y * front_scale + back_data.angular_velocity.y * back_scale;
            synced_data.angular_velocity.z = front_data.angular_velocity.z * front_scale + back_data.angular_velocity.z * back_scale;
            // 四元数插值有线性插值和球面插值，球面插值更准确，但是两个四元数差别不大是，二者精度相当
            // 由于是对相邻两时刻姿态插值，姿态差比较小，所以可以用线性插值
            synced_data.orientation.x = front_data.orientation.x * front_scale + back_data.orientation.x * back_scale;
            synced_data.orientation.y = front_data.orientation.y * front_scale + back_data.orientation.y * back_scale;
            synced_data.orientation.z = front_data.orientation.z * front_scale + back_data.orientation.z * back_scale;
            synced_data.orientation.w = front_data.orientation.w * front_scale + back_data.orientation.w * back_scale;
            // 线性插值之后要归一化
            synced_data.orientation.Normlize();
            current_imu_data_.push_back(synced_data);
            imu_data_buff_.pop_front();
            cloud_data_buff_.pop_front();
            return true;
        }

        if (imu_data_buff_.back().time < sync_time)
        {
            return false;
        }
        while (current_imu_data_.size() > 1)
        {
            current_imu_data_.pop_front();
        }
        while (imu_data_buff_.front().time < sync_time)
        {
            IMUData temp = imu_data_buff_.front();
            imu_data_buff_.pop_front();
            current_imu_data_.push_back(temp);
        }
        IMUData front_data = current_imu_data_.back();
        IMUData back_data = imu_data_buff_.at(0);
        IMUData synced_data;

        double front_scale = (back_data.time - sync_time) / (back_data.time - front_data.time);
        double back_scale = (sync_time - front_data.time) / (back_data.time - front_data.time);
        synced_data.time = sync_time;
        synced_data.linear_acceleration.x = front_data.linear_acceleration.x * front_scale + back_data.linear_acceleration.x * back_scale;
        synced_data.linear_acceleration.y = front_data.linear_acceleration.y * front_scale + back_data.linear_acceleration.y * back_scale;
        synced_data.linear_acceleration.z = front_data.linear_acceleration.z * front_scale + back_data.linear_acceleration.z * back_scale;
        synced_data.angular_velocity.x = front_data.angular_velocity.x * front_scale + back_data.angular_velocity.x * back_scale;
        synced_data.angular_velocity.y = front_data.angular_velocity.y * front_scale + back_data.angular_velocity.y * back_scale;
        synced_data.angular_velocity.z = front_data.angular_velocity.z * front_scale + back_data.angular_velocity.z * back_scale;
        // 四元数插值有线性插值和球面插值，球面插值更准确，但是两个四元数差别不大是，二者精度相当
        // 由于是对相邻两时刻姿态插值，姿态差比较小，所以可以用线性插值
        synced_data.orientation.x = front_data.orientation.x * front_scale + back_data.orientation.x * back_scale;
        synced_data.orientation.y = front_data.orientation.y * front_scale + back_data.orientation.y * back_scale;
        synced_data.orientation.z = front_data.orientation.z * front_scale + back_data.orientation.z * back_scale;
        synced_data.orientation.w = front_data.orientation.w * front_scale + back_data.orientation.w * back_scale;
        // 线性插值之后要归一化
        synced_data.orientation.Normlize();

        current_imu_data_.push_back(synced_data);
        cloud_data_buff_.pop_front();
            // std::cout << std::setprecision(12) << "current_imu_data_.time " << current_imu_data_.front().time 
            //     << "  " << current_imu_data_.back().time << std::endl;
        return true;
    }
    else
    {
        return false;
    }
}

bool LocalizationFlow::TransformData() {
    gnss_pose_ = Eigen::Matrix4f::Identity();

    current_gnss_data_.UpdateXYZ();
    gnss_pose_(0,3) = current_gnss_data_.local_E;
    gnss_pose_(1,3) = current_gnss_data_.local_N;
    gnss_pose_(2,3) = current_gnss_data_.local_U;
    gnss_pose_.block<3,3>(0,0) = current_imu_data_.back().GetOrientationMatrix();

    VelocityData temp = current_velocity_data_;
    temp.TransformCoordinate(lidar_to_imu_.inverse());
    distortion_adjust_ptr_->SetMotionInfo(0.1, temp);
    distortion_adjust_ptr_->AdjustCloud(current_cloud_data_.cloud_ptr, current_cloud_data_.cloud_ptr);

    return true;
}

bool LocalizationFlow::PublishData() {
    gnss_pub_ptr_->Publish(gnss_pose_, current_gnss_data_.time);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = state_.q.toRotationMatrix().cast<float>();
    pose.block<3, 1>(0, 3) = state_.p.cast<float>();
    laser_tf_pub_ptr_->SendTransform(pose, current_cloud_data_.time);
    laser_odom_pub_ptr_->Publish(pose, current_cloud_data_.time);
    current_scan_pub_ptr_->Publish(matching_ptr_->GetCurrentScan());
    SavePose(ground_truth_ofs_, gnss_pose_);
    SavePose(localization_ofs_, pose);
    return true;
}

bool LocalizationFlow::Filter()
{
    Predict();
    Correct();
    return true;
}

bool LocalizationFlow::Predict()
{
    Eigen::Vector3d pp = state_.p;
    Eigen::Vector3d vv = state_.v;
    Eigen::Quaterniond qq = state_.q;
    double w = 7.27220521664304e-05;
    Eigen::Vector3d gn(0, 0, -9.809432933396721);
    Eigen::Vector3d w_ie_n(0, w * std::cos(current_gnss_data_.latitude * M_PI / 180), 
    w * std::sin(current_gnss_data_.latitude * M_PI / 180));
    double rm = 6371829.37587;
    double rn = 6390325.45972;
    Eigen::Vector3d w_en_n(-vv[1] / (rm + current_gnss_data_.altitude), 
    vv[0] / (rn + current_gnss_data_.altitude), 
    vv[0] / (rn + current_gnss_data_.altitude) * std::tan(current_gnss_data_.latitude * M_PI / 180));
    Eigen::Vector3d w_in_n = w_ie_n + w_en_n;
    // std::cout << current_imu_data_.front().time << std::endl;
    for (int i = 1; i < current_imu_data_.size(); ++i)
    {
        double dt = current_imu_data_[i].time - current_imu_data_[i-1].time;
        Eigen::Vector3d wtemp = w_in_n * dt;
        double angle = wtemp.norm();
        Eigen::Quaterniond qn(1, 0, 0, 0);
        // qn = Eigen::Quaterniond(1, 0.5 * wtemp[0], 0.5 * wtemp[1], 0.5 * wtemp[2]);
        // qn.normalize();
        // std::cout << "qn " << qn.coeffs().transpose() << ", ";
        if (angle != 0)
        {
            wtemp = wtemp / angle;
            wtemp = std::sin(angle / 2) * wtemp;
            qn = Eigen::Quaterniond(std::cos(angle / 2), wtemp[0], wtemp[1], wtemp[2]);
        }
        // std::cout << "qn " << qn.coeffs().transpose() << std::endl;
        Eigen::Vector3d wb;
        wb [0] = 0.5 * current_imu_data_[i-1].angular_velocity.x +  0.5 * current_imu_data_[i].angular_velocity.x;
        wb [1] =  0.5 * current_imu_data_[i-1].angular_velocity.y +  0.5 * current_imu_data_[i].angular_velocity.y;
        wb [2] =  0.5 * current_imu_data_[i-1].angular_velocity.z +  0.5 * current_imu_data_[i].angular_velocity.z;
        wb = wb + state_.bg;
        wb = wb * dt;
        // std::cout << w_in_n.transpose() << "    " << wb.transpose() << std::endl;
        angle = wb.norm();
        Eigen::Quaterniond qb(1, 0, 0, 0);
        // qb = Eigen::Quaterniond(1, 0.5 * wb[0], 0.5 * wb[1], 0.5 * wb[2]);
        // qb.normalize();
        // std::cout << qb.coeffs().transpose() << ", ";
        if (angle != 0)
        {
            wb = wb / angle;
            wb = std::sin(angle / 2) * wb;
            qb = Eigen::Quaterniond(std::cos(angle / 2), wb[0], wb[1], wb[2]);
        }
        // std::cout << "qb " << qb.coeffs().transpose() << std::endl;
        Eigen::Quaterniond qq2 = qn.inverse() * qq * qb;
        Eigen::Vector3d f1(current_imu_data_[i-1].linear_acceleration.x, current_imu_data_[i-1].linear_acceleration.y,
        current_imu_data_[i-1].linear_acceleration.z);
        f1 = f1 + state_.ba;
        Eigen::Vector3d f2(current_imu_data_[i].linear_acceleration.x, current_imu_data_[i].linear_acceleration.y,
        current_imu_data_[i].linear_acceleration.z);
        f2 = f2 + state_.ba;
        Eigen::Vector3d vv2 = vv + dt * (0.5 * (qq * f1 + qq2 * f2) + gn);
        // std::cout << qq.inverse() * (0.5 * (qq * f1 + qq2 * f2) + gn) << std::endl;
        Eigen::Vector3d pp2 = pp + 0.5 * dt * (vv + vv2);
        pp = pp2;
        vv = vv2;
        qq = qq2;
    }
    // Eigen::Vector3d cvv(current_velocity_data_.linear_velocity.x, 
    // current_velocity_data_.linear_velocity.y, 
    // current_velocity_data_.linear_velocity.z);
    // std::cout << "Predict vv " << vv.transpose() << " " << (qq * cvv).transpose() << std::endl;
    // std::cout << "Predict vv2 " << (qq.inverse()*vv).transpose() << " " << cvv.transpose() << std::endl;
    // std::cout << "gnss " << gnss_pose_ << std::endl;
    // std::cout << "before " << state_.p.transpose() << std::endl;
    // std::cout << state_.q.toRotationMatrix() << std::endl;
    state_.p = pp;
    state_.v = vv;
    state_.q = qq;
    // std::cout << "after " << state_.p.transpose() << std::endl;
    // std::cout << state_.q.toRotationMatrix() << std::endl;

    Eigen::Matrix<double, 15, 15> Ft = Eigen::Matrix<double, 15, 15>::Zero();
    Ft.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> temp = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Vector3d ff(current_imu_data_.back().linear_acceleration.x, current_imu_data_.back().linear_acceleration.y,
    current_imu_data_.back().linear_acceleration.z);
    ff = qq * ff;
    temp(0, 1) = - ff[2];
    temp(0, 2) = ff[1];
    temp(1, 0) = ff[2];
    temp(1, 2) = -ff[0];
    temp(2, 0) = -ff[1];
    temp(2, 1) = ff[0];
    Ft.block<3, 3>(3, 6) = temp;
    Ft.block<3, 3>(3, 12) = qq.toRotationMatrix();
    temp.setZero();
    temp(0, 1) = w_ie_n(2);
    temp(0, 2) = -w_ie_n(1);
    temp(1, 0) = -w_ie_n(2);
    temp(2, 0) = w_ie_n(1);
    Ft.block<3, 3>(6, 6) = temp;
    Ft.block<3, 3>(6, 9) = -Ft.block<3, 3>(3, 12);
    Eigen::Matrix<double, 15, 6> Bt = Eigen::Matrix<double, 15, 6>::Zero();
    Bt.block<3, 3>(3, 3) = Ft.block<3, 3>(3, 12);
    Bt.block<3, 3>(6, 0) = Ft.block<3, 3>(6, 9);
    double T = current_imu_data_.back().time - current_imu_data_.front().time;
    Ft = Eigen::Matrix<double, 15, 15>::Identity() + Ft * T;
    Bt = Bt * T;
    Eigen::Matrix<double, 6, 1> W = Eigen::Matrix<double, 6, 1>::Zero();
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    Eigen::Vector3d noise_gyro(distribution(generator),distribution(generator),distribution(generator));
    Eigen::Vector3d noise_acc(distribution(generator),distribution(generator),distribution(generator));
    noise_gyro = noise_gyro * gyro_noise_;
    noise_acc = noise_acc * acc_noise_;
    W.head(3) = noise_gyro;
    W.tail(3) = noise_acc;
    error_state_.x = Ft * error_state_.x + Bt * W;
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Identity();
    Q.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * gyro_noise_ * gyro_noise_;
    Q.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * acc_noise_ * acc_noise_;
    error_state_.p = Ft * error_state_.p * Ft.transpose() + Bt * Q * Bt.transpose();
    return true;
}
bool LocalizationFlow::Correct()
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = state_.q.toRotationMatrix().cast<float>();
    pose.block<3, 1>(0, 3) = state_.p.cast<float>();
    Eigen::Matrix4f laser_odometry = pose * lidar_to_imu_;
    matching_ptr_->Update(current_cloud_data_, laser_odometry);
    laser_odometry = laser_odometry * lidar_to_imu_.inverse();
    // std::cout << "Correct before " << state_.p.transpose() << std::endl;
    // std::cout << "Correct before " << state_.q.coeffs().transpose() << std::endl;
    // std::cout << "Correct match " << laser_odometry.block<3, 1>(0, 3).cast<double>().transpose() << std::endl;
    // std::cout << "Correct match " << Eigen::Quaterniond(laser_odometry.block<3, 3>(0, 0).cast<double>()).coeffs().transpose() << std::endl;
    Eigen::Vector3d dp = pose.block<3, 1>(0, 3).cast<double>() - laser_odometry.block<3, 1>(0, 3).cast<double>();
    Eigen::Matrix<double, 3, 3> dr = pose.block<3, 3>(0, 0).cast<double>() * laser_odometry.block<3, 3>(0, 0).cast<double>().transpose();
    dr = dr - Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Vector3d dphi(dr(1, 2), dr(2, 0), dr(0, 1));
    Eigen::Matrix<double, 6, 1> Y;
    Y.head(3) = dp;
    Y.tail(3) = dphi;

    Eigen::Matrix<double, 6, 15> Gt = Eigen::Matrix<double, 6, 15>::Zero();
    Gt.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity();
    Gt.block<3, 3>(3, 6) = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 6, 6> Ct = Eigen::Matrix<double, 6, 6>::Identity();

    // Eigen::Matrix<double, 6, 1> N = Eigen::Matrix<double, 6, 1>::Zero();
    // std::random_device rd;
    // std::default_random_engine generator(rd());
    // std::normal_distribution<double> distribution(0.0, 1.0);
    // Eigen::Vector3d noise_dp(distribution(generator),distribution(generator),distribution(generator));
    // Eigen::Vector3d noise_dphi(distribution(generator),distribution(generator),distribution(generator));
    // noise_dp = noise_dp * dp_noise_;
    // noise_dphi = noise_dphi * dphi_noise_;
    // N.head(3) = noise_dp;
    // N.tail(3) = noise_dphi;
    Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity();
    R.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * dp_noise_ * dp_noise_;
    R.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * dphi_noise_ * dphi_noise_;

    Eigen::Matrix<double, 15, 6> K = error_state_.p * Gt.transpose()
    * (Gt * error_state_.p * Gt.transpose() + Ct * R * Ct.transpose()).inverse();
    error_state_.p = (Eigen::Matrix<double, 15, 15>::Identity() - K * Gt) * error_state_.p;
    // std::cout << " error dp1 " << error_state_.x.block<3, 1>(0, 0).transpose() << std::endl;
    error_state_.x = error_state_.x + K * (Y - Gt * error_state_.x);
    // std::cout << " error dp2 " << error_state_.x.block<3, 1>(0, 0).transpose() << std::endl;
    state_.p = state_.p - error_state_.x.block<3, 1>(0, 0);
    state_.v = state_.v - error_state_.x.block<3, 1>(3, 0);
    Eigen::Vector3d dphi_dir = error_state_.x.block<3, 1>(6, 0);
    double dphi_norm = dphi_dir.norm();
    if (dphi_norm != 0)
    {
        dphi_dir = dphi_dir / dphi_norm;
        dphi_dir = dphi_dir * std::sin(dphi_norm / 2);
    }
    Eigen::Quaterniond temp2(std::cos(dphi_norm / 2), dphi_dir[0], dphi_dir[1], dphi_dir[2]);
    state_.q = temp2 * state_.q;
    state_.bg = state_.bg - error_state_.x.block<3, 1>(9, 0);
    state_.ba = state_.ba - error_state_.x.block<3, 1>(12, 0);
    laser_odometry = Eigen::Matrix4f::Identity();
    laser_odometry.block<3, 3>(0, 0) = state_.q.toRotationMatrix().cast<float>();
    laser_odometry.block<3, 1>(0, 3) = state_.p.cast<float>();
    laser_odometry = laser_odometry * lidar_to_imu_;
    matching_ptr_->TransformCurrentScan(current_cloud_data_, laser_odometry);
    // std::cout << "Correct " << dphi_norm << " " << dp.norm() << std::endl;
    // std::cout << "ba " << state_.ba.transpose() << std::endl;
    // std::cout << "bg " << state_.bg.transpose() << std::endl;
    // std::cout << "dp " << dp.transpose() << std::endl;
    // std::cout << "Correct after " << state_.p.transpose() << std::endl;
    // std::cout << "Correct after " << state_.q.coeffs().transpose() << std::endl;
    // std::cout << " error dp " << error_state_.x.block<3, 1>(0, 0).transpose() << std::endl;
    // std::cout << "after2 " << state_.p.transpose() << std::endl;
    // std::cout << state_.q.toRotationMatrix() << std::endl;
    error_state_.x.setZero();
    return true;
}

bool LocalizationFlow::SavePose(std::ofstream& ofs, const Eigen::Matrix4f& pose) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            ofs << pose(i, j);
            
            if (i == 2 && j == 3) {
                ofs << std::endl;
            } else {
                ofs << " ";
            }
        }
    }

    return true;
}
}