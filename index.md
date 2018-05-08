By Karl Preisner and Joseph Schiarizzi
---
---

# Why
### Motivation
When people are in a temperature controlled environment, they are not always in agreement with what the temperature should be set at. A room that is 72 degrees Fahrenheit can feel hot to some people and cold to others. Most homes and buildings waste an enormous amount of energy (and money) overheating/cooling rooms. Additionally, there are a number of medical patients where their thermal comfort level is a crucial part of their treatment. A patient suffering from posttraumatic stress disorder (PTSD) is extremely dependent on their thermal comfort level.

### Long-term goal
A long-term solution to this problem is to develop a smart home thermostat that uses computer vison to automatically detect a person's thermal comfort level and adjust the room temperature accordingly. Computer vision coupled with machine learning are vastly growing fields of research that have the potential to predict a person's thermal comfort level. This smart thermostat will save home owners and buildings managers lots money. Why waste energy heating your whole house if you are only in one room at a time? The best part is that it is hands free!

Another useful product of thermal comfort prediction is the ability to create micro-controlled environments for the workplace. This can be envisioned as a worker's desk that has it's own heating/cooling vent and a camera that predicts the worker's thermal comfort.

### About our dataset
We compiled a dataset with roughly 2000+ color and thermal headshots of people. We used a Raspberry Pi camera (color) and a Flir Lepton 3 infrared camera (thermal) to take the images. The two cameras were positioned side-by-side, with the lenses roughly an inch apart. When we captured a headshot of a person, we saved 10 consecutive frames from both cameras. This gave us 10 color and 10 thermal images within the span of a second for each person. Additonally, when we surveyed a person, we asked them to fill out a short questionnaire asking what they think the temperature is, and what their thermal comfort level is with regard to the temperature. As a result, we have labeled data that can be used for machine learning to predict thermal comfort level. 

Below are samples of the raw color and thermal images that the camera system captured.
![Karl regular](https://i.imgur.com/6wLcmen.png)<br>
Karl Regular 1

![Karl ir](https://i.imgur.com/OuMFuOl.png) <br>
Karl IR

### Our goals for this class project
Our goals for this project are:
1. Extract the facial region of each image (both color and thermal)
2. Detect specific regions on the face that have the potential to be useful for predicting thermal comfort.

These two goals are small pieces of a long-term research project to predict the thermal comfort of a person.

# Related Work
There have been many studies on the Thermal Comfort of both people and animals.  Studies have used Thermal Comfort as an indicator of the observed person's happiness in their environment.  This research is important to office designers and car manufacturers, who want to maximize happiness of people in their spaces.

Recently, there have been more energy efficient buildings which rely on new methods for cooling/heating.  To test different systems effectiveness there has been [research using the building inhabitants' thermal comfort as an indicator of effectiveness](https://www.sciencedirect.com/science/article/pii/S0360132311002800).  Specifically in subtropical climates which tend to be hot and humid, like southeast Asia, the invention of air conditioning has [been called by the previous Prime Minister of Singapore](https://www.vox.com/2015/3/23/8278085/singapore-lee-kuan-yew-air-conditioning) "one of the signal inventions of history."  Because of this there is a significant interest in [research on optimizing the energy use and effectiveness of cooling systems in those areas](https://www.sciencedirect.com/science/article/pii/S0306261907001602).

# Approach
### Part 1

### Part 2
Detect the hottest features on the face and regions of interest.  Initially The goal was to line up checks, forehead, lips, etc. in the IR image and the regular image, but this was challenging and potentially not as useful.  Next, we tried blob detection to find the interest regions directly in the IR image and correlate their location to the regular image. This can be used to create a ML model for determining whether a person is comfortable in the given temperature or not from seeing a regular picture of them.

![small ir](https://i.imgur.com/NpZXZin.png)


# Implementations and Analysis
### Part 1
### Part 2
CV has a library for SimpleBlobDetection. It was challenging to run that on the IR images since they are so small and the blob detection wants to run on larger images. Running the blob detection on a regular image finds the part of the image that are most light.  We turn off shape detecting and choose intensity and size detection properties.

![regular blob](https://i.imgur.com/lcu9SUk.png)<br>
Less useful

![Karl face with red dots in ir](https://i.imgur.com/HYUTsag.png)<br>
More useful


Our points of interest are places where there is an especially dark blob in a light section or especially light blob in a dark section.  Being good insulators, the eyebrows seems to often get caught with a large detect, which could be better, but also makes sense. Aligning the key points must also be done a bit intelligently since a the faces are not perfectly aligned between the regular/IR images nor are the image proportions the same.

![final](https://i.imgur.com/fGQRBme.png)

# Conclusions

Joseph's conclusions:
- Formatting image data for different processes is hard.
- A model for guessing Thermal Comfort given an IR image seems plausible.
- A model for guessing Thermal Comfort given a regular image of a face seems less plausible.
