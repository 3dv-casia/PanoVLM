/*
 * @Author: Diantao Tu
 * @Date: 2022-09-06 12:02:06
 */
#ifndef _MATH_H_
#define _MATH_H_

/**
 * @description: 快速计算atan2，需要的时间约为std::atan2的三分之一，精度约为0.3度
 *              和opencv的FastAtan2相比会更快一点，精度一样
 * @param y 
 * @param x
 * @return 弧度(-pi, pi]
 */
template <typename T>
inline T FastAtan2(const T& y, const T& x)
{
    T ax = std::abs(x), ay = std::abs(y);
    T a = std::min(ax, ay)/(std::max(ax, ay) + std::numeric_limits<T>::epsilon() );
    T s = a*a;
    T r = ((-0.04432655554792128 * s + 0.1555786518463281) * s - 0.3258083974640975) * s * a + 0.9997878412794807 * a;
    if(ay > ax) 
        r = M_PI_2 - r;
    if(x < 0) 
        r = M_PI - r;
    if(y < 0) 
        r = -r;
    return r;
}

template<typename T>
inline T Square(const T& a)
{
    return a * a;
}

#endif