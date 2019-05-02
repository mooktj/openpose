#include <iostream>
#include <string>
#include <cstdio>
#include <fstream>
#include <vector>


std::vector<std::string> split(std::string s, std::string delimiter)
{
	size_t pos_start = 0, pos_end, delim_len = delimiter.length();
	std::string token;
	std::vector<std::string> res;
	while((pos_end = s.find(delimiter, pos_start)) != std::string::npos) 
	{
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}

	res.push_back(s.substr(pos_start));
	return res;
}

int main ()
{

	std::vector<std::vector<float>> data_all;
	for(int i = 0; i < 21; i++)
	{
		std::string file_name = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/sample_check/extracted_hm_" + std::to_string(i) + ".txt";
		std::ifstream filein(file_name);
		if(!filein) {
			std::cout << "Unable to open file\n";
			exit(1);
		}

		std::string line;
		std::vector<float> data;
		while(filein >> line)
		{
			// if(line == "]" || line == "[") continue;
			// std::cout << line << "\n";
			// data.push_back(std::stof(line,0));
			std::cout << line << "\n";
			// std::istringstream text(line);
			// std::vector<std::string> results((std::istream_iterator<WordDelimitedByComma>(text)),
			// 									std::istream_iterator<WordDelimitedByComma>());
			
			std::string str = line;
			std::string delimiter = ",";
			std::vector<std::string> v = split(str, delimiter);

			for(int i = 0; i < v.size(); i++)
			{
				if(i == v.size() - 1) continue;
				std::cout << v.at(i) << std::endl;
				data.push_back(std::stof(v.at(i),0));
			}

			// std::cout << "v size: " << v.size() << "\n";
			// std::cout << "v[last]: " << v[v.size() - 1] << "\n";

			// std::cout << "data size: " << data.size() << "\n";
		}

		// std::cout << "data size: " << data.size() << "\n";
		filein.close();
		data_all.push_back(data);

	}

	std::cout << "data _all size: " << data_all.size() << "\n";

	

	return 0;
}