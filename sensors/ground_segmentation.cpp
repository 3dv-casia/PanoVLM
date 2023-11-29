#include "ground_segmentation.h"

#include <chrono>
#include <cmath>
#include <list>
#include <memory>
#include <thread>
#include <pcl/io/pcd_io.h>

using namespace std::chrono_literals;

Bin::Bin() : min_z(std::numeric_limits<double>::max()), has_point_(false) {}

Bin::Bin(const Bin& bin) : min_z(std::numeric_limits<double>::max()),
                                           has_point_(false) {}

void Bin::addPoint(const pcl::PointXYZI& point) {
  const double d = sqrt(point.x * point.x + point.y * point.y);
  addPoint(d, point.z);
}

void Bin::addPoint(const double& d, const double& z) {
  has_point_ = true;
  if (z < min_z) {
    min_z = z;
    min_z_range = d;
  }
}

Bin::MinZPoint Bin::getMinZPoint() {
  MinZPoint point;

  if (has_point_) {
    point.z = min_z;
    point.d = min_z_range;
  }

  return point;
}


Segment::Segment(const unsigned int &n_bins,
				 const double &min_slope,
				 const double &max_slope,
				 const double &max_error,
				 const double &long_threshold,
				 const double &max_long_height,
				 const double &max_start_height,
				 const double &sensor_height) : bins_(n_bins),
												min_slope_(min_slope),
												max_slope_(max_slope),
												max_error_(max_error),
												long_threshold_(long_threshold),
												max_long_height_(max_long_height),
												max_start_height_(max_start_height),
												sensor_height_(sensor_height) {}

void Segment::fitSegmentLines()
{
	// Find first point.
	auto line_start = bins_.begin();
	while (!line_start->hasPoint())
	{
		++line_start;
		// Stop if we reached last point.
		if (line_start == bins_.end())
			return;
	}
	// Fill lines.
	bool is_long_line = false;
	double cur_ground_height = -sensor_height_;
	std::list<Bin::MinZPoint> current_line_points(1, line_start->getMinZPoint());
	LocalLine cur_line = std::make_pair(0, 0);
	for (auto line_iter = line_start + 1; line_iter != bins_.end(); ++line_iter)
	{
		if (line_iter->hasPoint())
		{
			Bin::MinZPoint cur_point = line_iter->getMinZPoint();
			if (cur_point.d - current_line_points.back().d > long_threshold_)
				is_long_line = true;
			if (current_line_points.size() >= 2)
			{
				// Get expected z value to possibly reject far away points.
				double expected_z = std::numeric_limits<double>::max();
				if (is_long_line && current_line_points.size() > 2)
				{
					expected_z = cur_line.first * cur_point.d + cur_line.second;
				}
				current_line_points.push_back(cur_point);
				cur_line = fitLocalLine(current_line_points);
				const double error = getMaxError(current_line_points, cur_line);
				// Check if not a good line.
				if (error > max_error_ ||
					std::fabs(cur_line.first) > max_slope_ ||
					(current_line_points.size() > 2 && std::fabs(cur_line.first) < min_slope_) ||
					is_long_line && std::fabs(expected_z - cur_point.z) > max_long_height_)
				{
					// Add line until previous point as ground.
					current_line_points.pop_back();
					// Don't let lines with 2 base points through.
					if (current_line_points.size() >= 3)
					{
						const LocalLine new_line = fitLocalLine(current_line_points);
						lines_.push_back(localLineToLine(new_line, current_line_points));
						cur_ground_height = new_line.first * current_line_points.back().d + new_line.second;
					}
					// Start new line.
					is_long_line = false;
					current_line_points.erase(current_line_points.begin(), --current_line_points.end());
					--line_iter;
				}
				// Good line, continue.
				else
				{
				}
			}
			else
			{
				// Not enough points.
				if (cur_point.d - current_line_points.back().d < long_threshold_ &&
					std::fabs(current_line_points.back().z - cur_ground_height) < max_start_height_)
				{
					// Add point if valid.
					current_line_points.push_back(cur_point);
				}
				else
				{
					// Start new line.
					current_line_points.clear();
					current_line_points.push_back(cur_point);
					float vertical_angle = atan(cur_point.z / cur_point.d) * 180 / M_PI;
					int scanID = 0;

					scanID = int((vertical_angle + 15) / 2 + 0.5);
					if (scanID <= 3 && scanID >= 0)
					{
						cur_ground_height = cur_point.z;
					}

					
				}
			}
		}
	}
	// Add last line.
	if (current_line_points.size() > 2)
	{
		const LocalLine new_line = fitLocalLine(current_line_points);
		lines_.push_back(localLineToLine(new_line, current_line_points));
	}
}

Segment::Line Segment::localLineToLine(const LocalLine &local_line,
									   const std::list<Bin::MinZPoint> &line_points)
{
	Line line;
	const double first_d = line_points.front().d;
	const double second_d = line_points.back().d;
	const double first_z = local_line.first * first_d + local_line.second;
	const double second_z = local_line.first * second_d + local_line.second;
	line.first.z = first_z;
	line.first.d = first_d;
	line.second.z = second_z;
	line.second.d = second_d;
	return line;
}

double Segment::verticalDistanceToLine(const double &d, const double &z)
{
	static const double kMargin = 0.1;
	double distance = -1;
	for (auto it = lines_.begin(); it != lines_.end(); ++it)
	{
		if (it->first.d - kMargin < d && it->second.d + kMargin > d)
		{
			const double delta_z = it->second.z - it->first.z;
			const double delta_d = it->second.d - it->first.d;
			const double expected_z = (d - it->first.d) / delta_d * delta_z + it->first.z;
			distance = std::fabs(z - expected_z);
		}
	}
	return distance;
}

double Segment::getMeanError(const std::list<Bin::MinZPoint> &points, const LocalLine &line)
{
	double error_sum = 0;
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		const double residual = (line.first * it->d + line.second) - it->z;
		error_sum += residual * residual;
	}
	return error_sum / points.size();
}

double Segment::getMaxError(const std::list<Bin::MinZPoint> &points, const LocalLine &line)
{
	double max_error = 0;
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		const double residual = (line.first * it->d + line.second) - it->z;
		const double error = residual * residual;
		if (error > max_error)
			max_error = error;
	}
	return max_error;
}

Segment::LocalLine Segment::fitLocalLine(const std::list<Bin::MinZPoint> &points)
{
	const unsigned int n_points = points.size();
	Eigen::MatrixXd X(n_points, 2);
	Eigen::VectorXd Y(n_points);
	unsigned int counter = 0;
	for (auto iter = points.begin(); iter != points.end(); ++iter)
	{
		X(counter, 0) = iter->d;
		X(counter, 1) = 1;
		Y(counter) = iter->z;
		++counter;
	}
	Eigen::VectorXd result = X.colPivHouseholderQr().solve(Y);
	LocalLine line_result;
	line_result.first = result(0);
	line_result.second = result(1);
	return line_result;
}

bool Segment::getLines(std::list<Line> *lines)
{
	if (lines_.empty())
	{
		return false;
	}
	else
	{
		*lines = lines_;
		return true;
	}
}

GroundSegmentation::GroundSegmentation(const GroundSegmentationParams &params) : 
										params_(params), segments_(params.n_segments, Segment(params.n_bins,
																							params.min_slope,
																							params.max_slope,
																							params.max_error_square,
																							params.long_threshold,
																							params.max_long_height,
																							params.max_start_height,
																							params.sensor_height))
{
}

void GroundSegmentation::segment(const PointCloud &cloud, std::vector<int>& ground_inlier)
{
	std::vector<int> segmentation;
	segmentation.resize(cloud.size(), 0);
	bin_index_.resize(cloud.size());
	segment_coordinates_.resize(cloud.size());
	resetSegments();
	insertPoints(cloud);
	for (unsigned int i = 0; i < params_.n_segments; ++i)
		segments_[i].fitSegmentLines();

	assignCluster(&segmentation);
	for(int i = 0; i < segmentation.size(); i++)
	{
		if(segmentation[i] == 1)
			ground_inlier.push_back(i);
	}

	if (false)
	{
		size_t n_ground = std::accumulate(segmentation.begin(), segmentation.end(), 0);
		// Visualize.
		pcl::PointCloud<pcl::PointXYZI>::Ptr obstacle_cloud(new PointCloud());
		obstacle_cloud->reserve(segmentation.size() - n_ground);
		// Get cloud of ground points.
		pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud(new PointCloud());
		ground_cloud->reserve(n_ground);
		for (size_t i = 0; i < cloud.size(); ++i)
		{
			if (segmentation.at(i) == 1)
				ground_cloud->push_back(cloud[i]);
			else
				obstacle_cloud->push_back(cloud[i]);
		}
		pcl::PointCloud<pcl::PointXYZI>::Ptr min_cloud(new PointCloud());
		getMinZPointCloud(min_cloud.get());
		pcl::io::savePCDFileASCII("obstacle.pcd", *obstacle_cloud);
		pcl::io::savePCDFileASCII("ground.pcd", *ground_cloud);
		pcl::io::savePCDFileASCII("min_cloud.pcd", *min_cloud);
		std::cout << "ground points : " << ground_cloud->size() << std::endl;
	}
}

void GroundSegmentation::getMinZPointCloud(PointCloud *cloud)
{
	cloud->reserve(params_.n_segments * params_.n_bins);
	const double seg_step = 2 * M_PI / params_.n_segments;
	double angle = -M_PI + seg_step / 2;
	for (auto seg_iter = segments_.begin(); seg_iter != segments_.end(); ++seg_iter)
	{
		for (auto bin_iter = seg_iter->begin(); bin_iter != seg_iter->end(); ++bin_iter)
		{
			const pcl::PointXYZI min = minZPointTo3d(bin_iter->getMinZPoint(), angle);
			cloud->push_back(min);
		}

		angle += seg_step;
	}
}

void GroundSegmentation::resetSegments()
{
	segments_ = std::vector<Segment>(params_.n_segments, Segment(params_.n_bins,
																 params_.min_slope,
																 params_.max_slope,
																 params_.max_error_square,
																 params_.long_threshold,
																 params_.max_long_height,
																 params_.max_start_height,
																 params_.sensor_height));
}

pcl::PointXYZI GroundSegmentation::minZPointTo3d(const Bin::MinZPoint &min_z_point,
												 const double &angle)
{
	pcl::PointXYZI point;
	point.x = cos(angle) * min_z_point.d;
	point.y = sin(angle) * min_z_point.d;
	point.z = min_z_point.z;
	return point;
}

void GroundSegmentation::assignCluster(std::vector<int> *segmentation)
{
	std::vector<std::thread> thread_vec(params_.n_threads);
	const size_t cloud_size = segmentation->size();
	for (unsigned int i = 0; i < params_.n_threads; ++i)
	{
		const unsigned int start_index = cloud_size / params_.n_threads * i;
		const unsigned int end_index = cloud_size / params_.n_threads * (i + 1);
		thread_vec[i] = std::thread(&GroundSegmentation::assignClusterThread, this,
									start_index, end_index, segmentation);
	}
	for (auto it = thread_vec.begin(); it != thread_vec.end(); ++it)
	{
		it->join();
	}
}

void GroundSegmentation::assignClusterThread(const unsigned int &start_index,
											 const unsigned int &end_index,
											 std::vector<int> *segmentation)
{
	const double segment_step = 2 * M_PI / params_.n_segments;
	for (unsigned int i = start_index; i < end_index; ++i)
	{
		Bin::MinZPoint point_2d = segment_coordinates_[i];
		const int segment_index = bin_index_[i].first;
		if (segment_index >= 0)
		{
			double dist = segments_[segment_index].verticalDistanceToLine(point_2d.d, point_2d.z);
			// Search neighboring segments.
			int steps = 1;
			while (dist == -1 && steps * segment_step < params_.line_search_angle)
			{
				// Fix indices that are out of bounds.
				int index_1 = segment_index + steps;
				while (index_1 >= params_.n_segments)
					index_1 -= params_.n_segments;
				int index_2 = segment_index - steps;
				while (index_2 < 0)
					index_2 += params_.n_segments;
				// Get distance to neighboring lines.
				const double dist_1 = segments_[index_1].verticalDistanceToLine(point_2d.d, point_2d.z);
				const double dist_2 = segments_[index_2].verticalDistanceToLine(point_2d.d, point_2d.z);
				// Select larger distance if both segments return a valid distance.
				if (dist_1 > dist)
				{
					dist = dist_1;
				}
				if (dist_2 > dist)
				{
					dist = dist_2;
				}
				++steps;
			}
			if (dist < params_.max_dist_to_line && dist != -1)
			{
				segmentation->at(i) = 1;
			}
		}
	}
}

void GroundSegmentation::getMinZPoints(PointCloud *out_cloud)
{
	const double seg_step = 2 * M_PI / params_.n_segments;
	const double bin_step = (sqrt(params_.r_max_square) - sqrt(params_.r_min_square)) / params_.n_bins;
	const double r_min = sqrt(params_.r_min_square);
	double angle = -M_PI + seg_step / 2;
	for (auto seg_iter = segments_.begin(); seg_iter != segments_.end(); ++seg_iter)
	{
		double dist = r_min + bin_step / 2;
		for (auto bin_iter = seg_iter->begin(); bin_iter != seg_iter->end(); ++bin_iter)
		{
			pcl::PointXYZI point;
			if (bin_iter->hasPoint())
			{
				Bin::MinZPoint min_z_point(bin_iter->getMinZPoint());
				point.x = cos(angle) * min_z_point.d;
				point.y = sin(angle) * min_z_point.d;
				point.z = min_z_point.z;

				out_cloud->push_back(point);
			}
			dist += bin_step;
		}
		angle += seg_step;
	}
}

void GroundSegmentation::insertPoints(const PointCloud &cloud)
{
	const double segment_step = 2 * M_PI / params_.n_segments;
	const double bin_step = (sqrt(params_.r_max_square) - sqrt(params_.r_min_square)) / params_.n_bins;
	const double r_min = sqrt(params_.r_min_square);
	for (unsigned int i = 0; i < cloud.size(); ++i)
	{
		const pcl::PointXYZI& point = cloud[i];
		const double range_square = point.x * point.x + point.y * point.y;
		const double range = sqrt(range_square);
		if (range_square < params_.r_max_square && range_square > params_.r_min_square)
		{
			const double angle = std::atan2(point.y, point.x);
			const unsigned int bin_index = (range - r_min) / bin_step;
			const unsigned int segment_index = (angle + M_PI) / segment_step;
			const unsigned int segment_index_clamped = segment_index == params_.n_segments ? 0 : segment_index;
			segments_[segment_index_clamped][bin_index].addPoint(range, point.z);
			bin_index_[i] = std::make_pair(segment_index_clamped, bin_index);
		}
		else
		{
			bin_index_[i] = std::make_pair<int, int>(-1, -1);
		}
		segment_coordinates_[i] = Bin::MinZPoint(range, point.z);
	}
}
