services:
  mysql:
    container_name: mongo_kie
    image: mongodb/mongodb-community-server
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: Ducky@123
      TZ: Asia/Hong_Kong
    volumes:
      - /home/ducky/mysql:/var/lib/mysql
      - ./my.cnf:/etc/mysql/my.cnf
    network_mode: host

