## seq2seq
基于此框架开发的模型包括：写诗/对联/聊天  
部署方式：通过AIPower自定义web服务(由于decoder采样无法直接通过serving部署)
训练步数：各20万步即可，过多会导致重复字生成
注意：模型文件为.model.ckpl,不带补数，注意处理

## uwsgi nginx command
uwsgi --ini uwsgi.ini  
uwsgi --stop uwsgi.pid  
uwsgi --reload uwsgi.pid  

cd /usr/local/nginx  
./sbin/nginx  
./sbin/nginx -s stop  
./sbin/nginx -s reload  


nginx配置，ubuntu位置在/etc/nginx/
nginx.conf
nginx 默认已经加入环境变量
```
server {
        listen       8111;
        server_name  127.0.0.1;

        location / {
            include uwsgi_params;
            uwsgi_pass 127.0.0.1:8001;
        }
        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
            root   html;
        }
    }
```

# 开发环境docker 运行相关
 nvidia-docker run  --name gpt-poet -u root -p 5000:5000 -p 8888:8888 -itd docker.oa.com:8080/public/tensorflow-cmb:1.14.0-gpu-py3.6 /bin/bash  
 nvidia-docker exec -u root -it gpt-poet /bin/bash 
 
 docker cp poet/ gpt-poet:/data/
 
# aipower环境安装
pip3 install uwsgi 
这样防止uwsgi找不到plugin的问题

