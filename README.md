1. Setup Zed SDK on PC (ZED_Explorer)
2. Made Apriltag board (already have, if you made new one please change config in config_board.json)
3. Captre images throgh SDK, redirect output dual-eye images path to ./cam*
4. Change the camera SN number based on the cam you used.
5. Run split_left.py to keep the left eye image and sort image into different folders (./cam0 ./cam1 ./cam2)
6. Run solve_extrinsics_from_board.py to get the extrinsic config
