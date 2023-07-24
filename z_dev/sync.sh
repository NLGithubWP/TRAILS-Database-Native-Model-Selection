

sshpass -p "xingnaili" scp ./TRAILS/Dockerfile xingnaili@panda.d2.comp.nus.edu.sg:/home/xingnaili/firmest_docker/TRAILS
sshpass -p "xingnaili" scp -r ./TRAILS/internal/pg_extension xingnaili@panda.d2.comp.nus.edu.sg:/home/xingnaili/firmest_docker/TRAILS/internal
sshpass -p "xingnaili" scp -r ./TRAILS/internal/ml/model_selec xingnaili@panda.d2.comp.nus.edu.sg:/home/xingnaili/firmest_docker/TRAILS/internal/ml/
#sshpass -p "xingnaili" scp -r ./TRAILS/internal xingnaili@panda.d2.comp.nus.edu.sg:/home/xingnaili/firmest_docker/TRAILS/


#sshpass -p "xingnaili" scp -r ./TRAILS/internal/pg_extension/src/lib.rs xingnaili@panda.d2.comp.nus.edu.sg:/home/xingnaili/firmest_docker/TRAILS/internal/pg_extension/src