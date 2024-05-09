#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <limits>

using namespace std;

struct Data
{
    vector<double> features;
    int label;
};

//Reads data from a csv file
void loadData(const string& filename, vector<Data> &sample) 
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Unable to open file " << filename << endl;
        return;
    }

    string line;
    while(getline(file,line))
    {
        stringstream ss(line);
        string token;
        Data inputData;

        while(getline(ss, token, ','))
        {
            inputData.features.push_back(stod(token));
        }

        inputData.label = static_cast<int>(inputData.features.back());
        inputData.features.pop_back();

        sample.push_back(inputData);
    }
}

// Function to calculate entropy for a set of samples
double calculateEntropy(const vector<Data>& data) 
{
    // Count the occurrences of each label
    map<int, int> labelCounts;
    for (const auto &d : data) 
    {
        labelCounts[d.label]++;
    }
    
    double entropy = 0.0;
    int total = data.size();
    for (const auto& pair : labelCounts) 
    {
        double probability = static_cast<double>(pair.second) / total;
        entropy -= probability * log2(probability);
    }

    return entropy;
}

// Function to calculate information gain for a given split point
double calculateInformationGain(const vector<Data>& data, int featureIndex, double splitPoint) 
{
    vector<Data> leftData, rightData;

    // Partition the data based on the split point
    for (const auto& d : data) 
    {
        if (d.features[featureIndex] < splitPoint) 
            leftData.push_back(d);
        else
            rightData.push_back(d);
    }

    // Calculate entropy of the entire dataset before the split
    vector<Data> dataBefore;
    for (const auto& d : data) 
    {
        dataBefore.push_back(d);
    }
    double entropyBefore = calculateEntropy(dataBefore);

    // Calculate the weighted average of the entropy after the split
    double weightLeft = static_cast<double>(leftData.size()) / data.size();
    double weightRight = static_cast<double>(rightData.size()) / data.size();
    double entropyLeft = calculateEntropy(leftData);
    double entropyRight = calculateEntropy(rightData);
    double weightedEntropyAfter = weightLeft * entropyLeft + weightRight * entropyRight;

    return entropyBefore - weightedEntropyAfter;
}

// Function to find the best split point and calculate the maximum information gain for a given feature
pair<double, double> findBestSplitAndInfoGain(const vector<Data>& data, int featureIndex) 
{
    // Extract the feature values for the given feature index and sort them
    vector<double> featureValues;
    for (const auto& d : data) 
    {
        featureValues.push_back(d.features[featureIndex]);
    }
    sort(featureValues.begin(), featureValues.end());
    
    // Initialize the best split and information gain
    pair<double, double> bestSplit = {numeric_limits<double>::lowest(), -1.0};
    
    // Iterate through the potential split points
    for (size_t i = 1; i < featureValues.size(); i++) 
    {
        double splitPoint = (featureValues[i - 1] + featureValues[i]) / 2.0;
        
        // Calculate information gain for the current split point
        double infoGain = calculateInformationGain(data, featureIndex, splitPoint);
        
        // Update the best split if the current info gain is higher
        if (infoGain > bestSplit.second)
            bestSplit = {splitPoint, infoGain};
    }
    
    return bestSplit;
}

int main() 
{
    vector<Data> data;
    loadData("small.data.csv", data);
    
    for (int i = 0; i < 10; i++) 
    {
        pair<double, double> bestSplit = findBestSplitAndInfoGain(data, i);
        
        cout << "Feature: " << i + 1 << endl;
        cout <<"Best Split Value: " << bestSplit.first << endl;
        cout << "Information Gain: " << bestSplit.second << " bits" << endl << endl;
    }

    return 0;
}