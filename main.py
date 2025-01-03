from detection import Detection
from prepare import Prepare
from clean import Clean

detection = Detection(cap_param='face.mp4')
detection.run()
preparation = Prepare(name='ana',num=detection.get_num())
preparation.prepare()
detection.training()
cleaning = Clean(filename='ana', limit=preparation.get_train_limit(), num=detection.get_num() ,result_directory=detection.get_result_directory())
cleaning.clean()