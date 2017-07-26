/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_COLOR_HPP
#define INSTRUMENT_GRAPH_COLOR_HPP


#include <ostream>
#include <sstream>


namespace Instrument {
	namespace Graph {
		struct Color {
			int _r, _g, _b;
			
			Color(int r, int g, int b)
				: _r(r), _g(g), _b(b)
			{
			}
			
			template <typename T>
			inline Color operator*(T factor)
			{
				Color result(_r * factor, _g * factor, _b * factor);
				result.saturate();
				return result;
			}
			
			inline Color operator+(Color const &other) const
			{
				Color result(_r + other._r, _g + other._g, _b + other._b);
				result.saturate();
				return result;
			}
			
			inline void saturate()
			{
				if (_r < 0) {
					_r = 0;
				} else if (_r > 255) {
					_r = 255;
				}
				if (_g < 0) {
					_g = 0;
				} else if (_g > 255) {
					_g = 255;
				}
				if (_b < 0) {
					_b = 0;
				} else if (_b > 255) {
					_b = 255;
				}
			}
			
			friend std::ostream &operator<<(std::ostream &os, Color const &color);
		};
		
		
		inline std::ostream &operator<<(std::ostream &os, Color const &color)
		{
			std::ostringstream oss;
			
			oss << "#";
			oss.width(2);
			oss.fill('0');
			oss << std::hex << color._r << color._g << color._b;
			
			os << oss.str();
			
			return os;
		}
		
		
		static inline Color getColor(int index)
		{
			// See: http://colorbrewer2.org/?type=qualitative&scheme=Set3&n=12
			switch (index) {
				case 0:
					return Color(141,211,199);
				case 1:
					return Color(255,255,179);
				case 2:
					return Color(190,186,218);
				case 3:
					return Color(251,128,114);
				case 4:
					return Color(128,177,211);
				case 5:
					return Color(253,180,98);
				case 6:
					return Color(179,222,105);
				case 7:
					return Color(252,205,229);
				case 8:
					return Color(217,217,217);
				case 9:
					return Color(188,128,189);
				case 10:
					return Color(204,235,197);
				case 11:
					return Color(255,237,111);
				default:
					return Color(255, 255, 255);
			}
		}
		
		
		static inline Color getBrightColor(int index, float brightness=0.5)
		{
			Color color = getColor(index);
			Color white(255, 255, 255);
			
			return (color * (1.0 - brightness)) + (white * brightness);
		}
		
		
		static inline Color getDarkColor(int index, float darkness=0.5)
		{
			Color color = getColor(index);
			
			color._r = (color._r - (255*darkness)) / (1.0 - darkness);
			color._g = (color._g - (255*darkness)) / (1.0 - darkness);
			color._b = (color._b - (255*darkness)) / (1.0 - darkness);
			color.saturate();
			
			return color;
		}
		
	}
}


#endif
