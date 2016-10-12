0925
from 9am -- 8:30 pm, run script_rpn_pedestrian_VGG16_caltech
the results are in "external/code3.2.1/results" and  "external/code3.2.1/data-USA/res"

0929-1001 RPN + BF experiment: all stages are completed, except the last plotting in dbEval_RPNBF
=====log==========
........ 
Test: 4022 / 4024 
miss rate:9.82
Loading detections: /usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian/external/code3.2.1/results/UsaTest
	Algorithm #1: RPN-ped
	Algorithm #2: RPN+BF
Loading ground truth: /usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian/external/code3.2.1/results/UsaTest
	Experiment #1: Reasonable
Evaluating: /usr/local/data/yuguang/git_all/RPN_BF_pedestrain/RPN_BF-RPN-pedestrian/external/code3.2.1/results/UsaTest
	Exp 1/1, Alg 1/2: Reasonable/RPN-ped
	Exp 1/1, Alg 2/2: Reasonable/RPN+BF
Error using figure
This functionality is no longer supported under the -nojvm startup option. For
more information, see "Changes to -nojvm Startup Option" in the MATLAB Release
Notes. To view the release note in your system browser, run
web('http://www.mathworks.com/help/matlab/release-notes.html#btsurqv-6',
'-browser').

Error in dbEval_RPNBF>plotExps (line 209)
  figure(1); clf; grid on; hold on; n=length(xs1); h=zeros(1,n);

Error in dbEval_RPNBF (line 125)
  plotExps( res, plotRoc, plotAlg, plotNum, plotName, ...

Error in script_rpn_bf_pedestrian_VGG16_caltech (line 193)
dbEval_RPNBF;

Error in run (line 96)
evalin('caller', [script ';']);


% ================ 1007: compare recall rate after NMS
gt recall rate = 0.6276
gt recall rate after nms-1 = 0.6168
gt recall rate after nms-2 = 0.5689
gt recall rate after nms-3 = 0.5547

% ================ 1010
training set (wider e1-e3) statistics before BF [before NMS: -1, nms thresh: 0.7, after nms: 1000]
score_threshold:0.006636
gt recall rate (ol >0.5) = 0.5928
gt recall rate (ol >0.7) = 0.2717
gt recall rate (ol >0.8) = 0.1001
gt recall rate (ol >0.9) = 0.0090
gt_num: 23055

