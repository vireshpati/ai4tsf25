ssh -L 60281:localhost:60281 vpati3@keep.isye.gatech.edu \
    -t "ssh -L 60281:localhost:60281 compute01 \
    -t 'ssh -L 60281:localhost:60281 isye-syang605'"