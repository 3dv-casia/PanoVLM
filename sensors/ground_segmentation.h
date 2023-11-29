#ifndef GROUND_SEGMENTATION_H_
#define GROUND_SEGMENTATION_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <list>

struct GroundSegmentationParams {
  GroundSegmentationParams() :
      visualize(false),
      r_min_square(0.3 * 0.3),
      r_max_square(20*20),
      n_bins(30),
      n_segments(180),
      max_dist_to_line(0.15),
      min_slope(0),
      max_slope(1),
      n_threads(4),
      max_error_square(0.01),
      long_threshold(2.0),
      max_long_height(0.1),
      max_start_height(0.2),
      sensor_height(0.2),
      line_search_angle(0.2) {}

  // Visualize estimated ground.
  bool visualize;
  // Minimum range of segmentation.
  double r_min_square;
  // Maximum range of segmentation.
  double r_max_square;
  // Number of radial bins.
  int n_bins;
  // Number of angular segments.
  int n_segments;
  // Maximum distance to a ground line to be classified as ground.
  double max_dist_to_line;
  // Min slope to be considered ground line.
  double min_slope;
  // Max slope to be considered ground line.
  double max_slope;
  // Max error for line fit.
  double max_error_square;
  // Distance at which points are considered far from each other.
  double long_threshold;
  // Maximum slope for
  double max_long_height;
  // Maximum heigh of starting line to be labelled ground.
  double max_start_height;
  // Height of sensor above ground.
  double sensor_height;
  // How far to search for a line in angular direction [rad].
  double line_search_angle;
  // Number of threads.
  int n_threads;
};

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

typedef std::pair<pcl::PointXYZI, pcl::PointXYZI> PointLine;


class Bin {
public:
  struct MinZPoint {
    MinZPoint() : z(0), d(0) {}
    MinZPoint(const double& d, const double& z) : z(z), d(d) {}
    bool operator==(const MinZPoint& comp) {return z == comp.z && d == comp.d;}

    double z;
    double d;
  };

private:

  std::atomic<bool> has_point_;
  std::atomic<double> min_z;
  std::atomic<double> min_z_range;

public:

  Bin();

  /// \brief Fake copy constructor to allow vector<vector<Bin> > initialization.
  Bin(const Bin& segment);

  void addPoint(const pcl::PointXYZI& point);

  void addPoint(const double& d, const double& z);

  MinZPoint getMinZPoint();

  inline bool hasPoint() {return has_point_;}

};

class Segment {
public:
  typedef std::pair<Bin::MinZPoint, Bin::MinZPoint> Line;

  typedef std::pair<double, double> LocalLine;

private:
  // Parameters. Description in GroundSegmentation.
  const double min_slope_;
  const double max_slope_;
  const double max_error_;
  const double long_threshold_;
  const double max_long_height_;
  const double max_start_height_;
  const double sensor_height_;

  std::vector<Bin> bins_;

  std::list<Line> lines_;

  LocalLine fitLocalLine(const std::list<Bin::MinZPoint>& points);

  double getMeanError(const std::list<Bin::MinZPoint>& points, const LocalLine& line);

  double getMaxError(const std::list<Bin::MinZPoint>& points, const LocalLine& line);

  Line localLineToLine(const LocalLine& local_line, const std::list<Bin::MinZPoint>& line_points);


public:

  Segment(const unsigned int& n_bins,
          const double& min_slope,
          const double& max_slope,
          const double& max_error,
          const double& long_threshold,
          const double& max_long_height,
          const double& max_start_height,
          const double& sensor_height);

  double verticalDistanceToLine(const double& d, const double &z);

  void fitSegmentLines();

  inline Bin& operator[](const size_t& index) {
    return bins_[index];
  }

  inline std::vector<Bin>::iterator begin() {
    return bins_.begin();
  }

  inline std::vector<Bin>::iterator end() {
    return bins_.end();
  }

  bool getLines(std::list<Line>* lines);

};

class GroundSegmentation {

  const GroundSegmentationParams params_;

  // Access with segments_[segment][bin].
  std::vector<Segment> segments_;

  // Bin index of every point.
  std::vector<std::pair<int, int> > bin_index_;

  // 2D coordinates (d, z) of every point in its respective segment.
  std::vector<Bin::MinZPoint> segment_coordinates_;

  void assignCluster(std::vector<int>* segmentation);

  void assignClusterThread(const unsigned int& start_index,
                           const unsigned int& end_index,
                           std::vector<int>* segmentation);

  void insertPoints(const PointCloud& cloud);


  void getMinZPoints(PointCloud* out_cloud);

  pcl::PointXYZI minZPointTo3d(const Bin::MinZPoint& min_z_point, const double& angle);

  void getMinZPointCloud(PointCloud* cloud);

  void resetSegments();

public:

  GroundSegmentation(const GroundSegmentationParams& params = GroundSegmentationParams());

  void segment(const PointCloud& cloud, std::vector<int>& ground_inlier);

};

#endif // GROUND_SEGMENTATION_H_
