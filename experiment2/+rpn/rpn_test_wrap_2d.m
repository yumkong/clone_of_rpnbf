function feat_list = rpn_test_wrap_2d(conf, model_stage, imdb)
    cache_dir = fullfile(pwd, 'output2', conf.exp_name, 'rpn_cache', model_stage.cache_name, imdb.name);
    % load test roidb data which is already made at training time
    fprintf('Preparing test roidb data...\n');
    cache_roidb_dir = fullfile(pwd, 'output2', conf.exp_name, 'rpn_cache', model_stage.cache_name, 'WIDERFACE_train');
    test_roi_name = fullfile(cache_roidb_dir, 'test_input_roidb_all.mat');
    try
        load(test_roi_name); %image_roidb_val
    catch
        fprintf('Error loading test roidb !!!\n');
    end
    %aboxes                      = helper.rpn_test(conf, imdb, cache_dir, ...
    [feat_list, label_list]                      = rpn.rpn_test_2d(conf, imdb, image_roidb_val, cache_dir, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name, ...
                                        'three_scales',     false, ...
                                        'feat_per_img',     10); %0124: root switch of test with 1 scale or 3 scales
    % draw 2D distribution plot
	figure(6);
    % liu: scatter plot by group
    gscatter(feat_list(:,1), feat_list(:,2), label_list,'br','xo');
    set(gca, 'FontSize', 12);
    xlabel('Feature\_1','FontSize', 15, 'FontWeight','bold');
    ylabel('Feature\_2','FontSize', 15, 'FontWeight','bold');
    %0810 plot two centers
    if 1
    hold on
    plot(0.0012, 0.0182,'-s','MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','k')
    plot(0.0002, 0.0682,'-s','MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor','k')
    hold off
    end
    %blue cross -- non-face, red circle-- face
    legend('\fontsize{12} Non-Face','\fontsize{12} Face');
    %%%% save plot %%
    savefile = fullfile(cache_dir, '2Dplot');
    export_fig(savefile, '-png', '-a1', '-native');
end

