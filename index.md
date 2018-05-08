By Karl Preisner and Joseph Schiarizzi

# Why
### Real Problem
Sometimes people are hot and people are cold so they change the temperature in the whole room. But it would be great if we had micro-climates around each person in a space. We would have happier workers because not everyone can always agree on the temperature.  We would also use less energy to cool/heat smaller areas, around someone's desk for example, rather than an entire room.

The process of deciding the temperature should be automated, which requires a computer to understand whether the human is comfortable or not.

![Karl regular]()<br>
Karl Regular 1

![Karl]() <br>
Karl IR

This is an older picture of Karl.

Karl 2 IR



### Goal
Take a thermal image of the user and determine if they like the temperature or not.

# Related Work
There have been many studies on the Thermal Comfort of both people and animals.  Studies have used Thermal Comfort as an indicator of the observed person's happiness in their environment.  This is potentially important to office designers and car manufacturers, who want to maximize happiness of people in their spaces.

Recently, there have been more energy efficient buildings which rely on new methods for cooling/heating.  To test different systems effectiveness there has been [research using the building inhabitants' thermal comfort as an indicator of effectiveness](https://www.sciencedirect.com/science/article/pii/S0360132311002800).  Specifically in subtropical climates which tend to be hot and humid, like southeast Asia, the invention of air conditioning has [been called by the previous Prime Minister of Singapore](https://www.vox.com/2015/3/23/8278085/singapore-lee-kuan-yew-air-conditioning) "one of the signal inventions of history."  Because of this there is a significant interest in [research on optimizing the energy use and effectiveness of cooling systems in those areas](https://www.sciencedirect.com/science/article/pii/S0306261907001602).

# Approach
### Part 1

### Part 2
Detect the hottest features on the face and regions of interest.  Initially The goal was to line up checks, forehead, lips, etc. in the IR image and the regular image, but this was challenging and potentially not as useful.  Next we tried blob detection to find the interest regions directly in the IR image and correlate their location to the regular image. This could be used to create a ML model for determining whether a person is comfortable in the given temperature or not from seeing a regular picture of them.

![small ir](https://i.imgur.com/NpZXZin.png)


# Implementations and Analysis
### Part 1
### Part 2
CV has a library for SimpleBlobDetection. It was challenging to run that on the IR images since they are so small and the blob detection wants to run on larger images. Running the blob detection on a regular image finds the part of the image that are most light.  We turn off shape detecting and choose intensity and size detection properties.

![regular blob](https://i.imgur.com/lcu9SUk.png)<br>
Less useful

![Karl face with red dots in ir](https://i.imgur.com/HYUTsag.png)<br>
More useful


Our points of interest are places where there is an especially dark blob in a light section or especially light blob in a dark section.  Being good insulators, the eyebrows seems to always get caught with a large detect, which could be better.

# Conclusions

Joseph's conclusions:
- Formatting image data for different processes is hard.
- A model for guessing Thermal Comfort given an IR image seems plausible.
- A model for guessing Thermal Comfort given a regular image of a face seems less plausible.
