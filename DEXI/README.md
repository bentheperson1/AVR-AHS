# Node RED custom node development for DEXI

Make sure you're connected to DEXI's wifi network!

1. Launch the container from the base node-red image 
`docker run -it -p 1880:1880 -v ${PWD}/flows:/data -v ${PWD}/node-red-dexi:/node-red-dexi --name dexi-node-red nodered/node-red:latest-minimal`

2. Connect to the container
`docker exec -it dexi-node-red /bin/bash`

3. Install the nodes for local development since we already mapped the host folder node-red-dexi to the container folder /node-red-dexi
`npm install /node-red-dexi`

