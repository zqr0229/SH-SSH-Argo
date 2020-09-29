%比较aviso的时间相近的对比sla adt与sh
load  ('E:\Python\sh-adt-argo\0.10.1\SH_test.mat')
diff_sla=[]
diff_adt=[]
diff_sh=[]
argotime=[];
avisotime=[];
argolat=[];
argolon=[];
avisolat=[];
avisolon=[];
id=[];
for i =1:length(SH)
    for i_compare=i:length(SH)
        if abs(realdata_avisotime(i)-realdata_avisotime(i_compare))<0.1 & realdata_id(i)~=realdata_id(i_compare)
            realdata_avisotime(i)-realdata_avisotime(i_compare)      
            d_sla=realdata_sla(i)-realdata_sla(i_compare)
            d_adt=realdata_adt(i)-realdata_adt(i_compare)
            d_sh=SH(i)-SH(i_compare)
            diff_sla=[diff_sla,d_sla]
            diff_adt=[diff_adt,d_adt]
            diff_sh=[diff_sh,d_sh]
            argotime=[argotime,realdata_argotime(i),realdata_argotime(i_compare)];
            avisotime=[avisotime,realdata_avisotime(i),realdata_avisotime(i_compare)];
            argolat=[argolat,realdata_argolat(i),realdata_argolat(i_compare)];
            argolon=[argolon,realdata_argolon(i),realdata_argolon(i_compare)];
            avisolat=[avisolat,realdata_avisolat(i),realdata_avisolat(i_compare)];
            avisolon=[avisolon,realdata_avisolon(i),realdata_avisolon(i_compare)];
            id=[id,realdata_id(i),realdata_id(i_compare)];
        end
    end
    
end
save ('E:\Python\sh-adt-argo\0.10.1\SH_adtsla_test.mat','diff_sla','diff_adt','diff_sh','argotime','avisotime','argolat','argolon','avisolat','avisolon','id')