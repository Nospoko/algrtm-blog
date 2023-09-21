# Use the official Node.js image as our base image
FROM node:18.17.1

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install hexo-cli globally so we can run hexo commands
RUN npm install -g hexo-cli

# Expose port 4000 for Hexo server
EXPOSE 4000

# By default, start the Hexo server
CMD ["hexo", "server"]
