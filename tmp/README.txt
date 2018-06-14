Zip에 대한 설명
1. create_fddb_tf_record.py
  FDDB 데이터베이스에 대한 어뎁터입니다. 
  tensorflow가 데이터를 입력받는 형식 (.record)에 맞게 데이터를 변형하였습니다. 
  주의할 점 하나가 있는데 FDDB는 본래 ellipse형태로 얼굴을 annotate를 해놨는데 그거를 모두 정사각형으로 바꿔놨습니다. 바꾼형식은 .py 파일안에 convert_ellipse2rect()라는 펑션에 있습니다.

2. ssd_mobilenet_v1_face.config
  텐서플로우 모델 깃헙을 클론해보니 이미 object_detection에 관한 모델들이 다양하게 있었습니다.
  첫 시작으로는 원래 있었던 컨피그 파일을 살짝 변형하여 트레이닝을 해보는게 어떻게나 생각을 해봤습니다.
  결과적으로 저 컨피그 파일은 원래 제공되는 ssd_mobilenet_v1_pet.config와 거의 동일하다고 보면 될것같습니다. 다만 저희의 아키텍쳐에는 '얼굴만' 디텍를 하는것이기 때문에 number of classes를 1로 설정을 해놨습니다.

피드백 주시면 감사하겠습니다!


----UPDATE 6/14/2018 - 10:15pm
1. create_fddb_tf_record.py
  박스치는 구역을 살짝 변경하였습니다. 박스를 치는 과정에서 이미지를 벗어나는 일이 있었습니다.

2. create_fddb_tf_record_test.py
  박스를 제대로 치는지 테스트 해볼수 있는 간단한 스크립트 입니다. 제가 샘플 몇개를 tmp/sample_faces 라는 폴더 안에 넣어놨습니다.

----UPDATE 6/14/2018 - 10:48pm
1. create_fddb_tf_record.py
  바이트 인코딩이 필요한 항목에 대해서 utf-8 인코딩을 추가하였습니다.

