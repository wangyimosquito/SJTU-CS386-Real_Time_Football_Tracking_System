# SJTU-CS386-football-tracking
SJTU - CS386 Course Project, Football Player Tracking System

# Readme
### GUI example
![avatar](/img/GUI.png)

### Install

Dependencies

+ numpy
+ opencv3
+ PyQt5
+ matplotlib



### Usage

##### Run the testcase

The testcase in this repository is `vid/panorama.avi`。To run the tracking system, click `exe/main.exe`，then click `Play` button for real-time tracking and analysis.

This system has no pause function for it is designed to be a real-time tracking system. The backend won't store previous analysis results。But in the `txt/posistion data` preserves the 20 players' position records in the testcase video.

##### Quit the System

Click the `Quit` button for quiting. Close the window directly will cause the next running of this system to preserve the last running speed, running distance and running map records. But if you meant to preserve these, click the close button at up right.

##### Export the Data

For manually selected player technical index records, after clicking the `Export Data` button, `txt/playerAction.txt` data records will be automatically generated. The image of the running distance of the players and the average running speed of the players are stored in `project file/src/Speed.png` and `src/Distance.png`, and the running path images of the players of the two teams are stored in `img/HotMapTeam1. png` and `img/HotMapTeam2.png`.

##### Change video source

The game video source of this project can be replaced. The replacement procedure is as follows:

1. Run the old game video again, select the `Clear Above Data` button on the central data control panel to clear the data records related to the old video, and click the `Quit` button to exit.

2. Check whether the records in the `txt` file are cleared. If not, you can choose to delete them manually.

3. Check whether the side-view image is cleared under the `img` file, if not, you can choose to delete it manually.

4. Comment out the following interactive interface running code in `src/main.py`.

   ```python
   import playertrack
   from pyqtMultithread2 import mainwin
   ...
   app = QtWidgets.QApplication(sys.argv)
   w = mainwin()
   w.show()
   sys.exit(app.exec_())
   ```

5. Put the source file you want to run in `vid` and name it `panorama.avi`.

6. Run the main.py file in the terminal. At this time, the script will perform background extraction, manual calibration of the four corners of the field, manual calibration of player colors, and manual frame selection of 20 player positions.

    + Background extraction: The script automatically extracts the average background of the first 500 frames of the video.
    + Manual calibration of the four corners of the arena: click the four corners of the arena in the order of lower left, upper left, upper right, and lower right, and press any key to exit after selection.
    + Player color calibration: Select the most obvious part of the torso of the players of the two teams and the goalkeeper, and click it. After the selection is complete, press any key to exit.
    + Manual frame selection of 20 player positions: manually select the initial positions of 20 players, the top 10 players form a team, and the bottom 10 players form a team. After manually selecting a frame, press any key, and the frame will be displayed on the screen. After the 20 players are selected, press any key to exit.

7. After the above preparatory work is completed, you can restore the commented code and run the interactive interface again.
