@echo off
REM NEXTBASE512G finepixXP30 GoProHero4 s5690 GoProHero5
REM GoProHero5PhotoMedium, GoProHero5PhotoWide, GoProHero5PhotoNarrow

SET hero5ch=C:/Users/GRAHAM~1/OneDrive/DOCUME~1/PHD/images/bass/fiducial/charter/GOPROH~2
SET hero5sh=C:/Users/GRAHAM~1/OneDrive/DOCUME~1/PHD/images/bass/fiducial/shore/GOPROH~2

SET hero4ch=C:/Users/GRAHAM~1/OneDrive/DOCUME~1/PHD/images/bass/fiducial/charter/GOPROH~1
SET hero4sh=C:/Users/GRAHAM~1/OneDrive/DOCUME~1/PHD/images/bass/fiducial/shore/GOPROH~1

REM Charter Wide
C:\development\python\opencvlib\lenscorrection\lenscorrection.py -m undistort_fisheye -c GoProHero5PhotoWide -o %hero5ch%/undistorted_fisheye -p %hero5ch%
REM Shore Wide
C:\development\python\opencvlib\lenscorrection\lenscorrection.py -m undistort_fisheye -c GoProHero5PhotoWide -o %hero5sh%/undistorted -p %hero5sh%
