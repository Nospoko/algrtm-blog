```sh
docker build -t algrtm-blog .
docker run -it --rm -v $(pwd):/usr/src/app -p 4000:4000 --entrypoint bash algrtm-blog

# Inside container
cd blog
hexo server
```
