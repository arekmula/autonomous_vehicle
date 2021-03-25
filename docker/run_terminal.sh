ID=`docker container ls | cut -d" " -f1 | grep -oP "\b[a-z0-9]{12}\b"`
echo $ID
docker exec -it $ID bash
