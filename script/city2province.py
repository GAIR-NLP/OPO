city_list = {
  "北京": ["北京"],
  "天津": ["天津"],
  "山西": ["太原", "阳泉", "晋城", "长治", "临汾", "运城", "忻州", "吕梁", "晋中", "大同", "朔州"],
  "河北": ["沧州", "石家庄", "唐山", "保定", "廊坊", "衡水", "邯郸", "邢台", "张家口", "辛集", "秦皇岛", "定州", "承德", "涿州",
           "丰宁满族自治县", "大厂回族自治县", "孟村回族自治县","宽城满族自治县", "青龙满族自治县", "围场满族蒙古族自治县"],
  "山东": ["济南", "淄博", "聊城", "德州", "滨州", "济宁", "菏泽", "枣庄", "烟台", "威海", "泰安", "青岛", "临沂", "东营", "潍坊",
           "日照"],
  "河南": ["郑州", "新乡", "鹤壁", "安阳", "焦作", "濮阳", "开封", "驻马店", "商丘", "三门峡", "南阳", "洛阳", "周口", "许昌",
           "信阳", "漯河", "平顶山", "济源"],
  "广东": ["珠海", "中山", "肇庆", "深圳", "清远", "揭阳", "江门", "惠州", "河源", "广州", "佛山", "东莞", "潮州", "汕尾", "梅州",
           "阳江", "云浮", "韶关", "湛江", "汕头", "茂名", "乳源瑶族自治县", "连南瑶族自治县", "连山壮族瑶族自治县"],
  "浙江": ["舟山", "温州", "台州", "绍兴", "衢州", "宁波", "丽水", "金华", "嘉兴", "湖州", "杭州", "景宁畲族自治县"],
  "宁夏": ["中卫", "银川", "吴忠", "石嘴山", "固原"],
  "江苏": ["镇江", "扬州", "盐城", "徐州", "宿迁", "无锡", "苏州", "南通", "南京", "连云港", "淮安", "常州", "泰州"],
  "湖南": ["长沙", "邵阳", "怀化", "株洲", "张家界", "永州", "益阳", "湘西", "娄底", "衡阳", "郴州", "岳阳", "常德", "湘潭",
           "城步苗族自治县", "新晃侗族自治县","江华瑶族自治县", "芷江侗族自治县", "通道侗族自治县", "靖州苗族侗族自治县", "麻阳苗族自治县"],
  "吉林": ["长春", "通化", "松原", "四平", "辽源", "吉林", "延边", "白山", "白城", "伊通满族自治县", "前郭尔罗斯蒙古族自治县",
           "长白朝鲜族自治县"],
  "福建": ["漳州", "厦门", "福州", "三明", "莆田", "宁德", "南平", "龙岩", "泉州"],
  "甘肃": ["张掖", "陇南", "兰州", "嘉峪关", "白银", "武威", "天水", "庆阳", "平凉", "酒泉", "金昌", "定西", "甘南", "临夏",
           "天祝藏族自治县", "肃北蒙古族自治县", "东乡族自治县", "张家川回族自治县", "肃南裕固族自治县", "阿克塞哈萨克族自治县",
           "积石山保安族东乡族撒拉族自治县"],
  "陕西": ["榆林", "西安", "延安", "咸阳", "渭南", "铜川", "商洛", "汉中", "宝鸡", "安康"],
  "辽宁": ["营口", "铁岭", "沈阳", "盘锦", "辽阳", "锦州", "葫芦岛", "阜新", "抚顺", "丹东", "大连", "朝阳", "本溪", "鞍山",
           "喀喇沁左翼蒙古族自治县", "阜新蒙古族自治县", "岫岩满族自治县", "新宾满族自治县", "桓仁满族自治县", "清原满族自治县",
           "本溪满族自治县", "宽甸满族自治县"],
  "江西": ["鹰潭", "宜春", "上饶", "萍乡", "南昌", "景德镇", "吉安", "抚州", "新余", "九江", "赣州"],
  "黑龙江": ["伊春", "七台河", "牡丹江", "鸡西", "黑河", "鹤岗", "哈尔滨", "绥化", "双鸭山", "齐齐哈尔", "佳木斯", "大庆", "大兴安岭",
             "杜尔伯特蒙古族自治县"],
  "安徽": ["宣城", "铜陵", "六安", "黄山", "淮南", "合肥", "阜阳", "亳州", "安庆", "池州", "宿州", "芜湖", "马鞍山", "淮北", "滁州",
           "蚌埠"],
  "湖北": ["孝感", "武汉", "十堰", "荆门", "黄冈", "襄阳", "咸宁", "随州", "黄石", "荆州", "宜昌", "鄂州", "恩施", "天门", "潜江",
           "神农架", "仙桃", "五峰土家族自治县", "长阳土家族自治县"],
  "青海": ["西宁", "海西", "海东", "玉树", "黄南", "海北", "果洛", "互助土族自治县", "化隆回族自治县", "大通回族土族自治县",
           "循化撒拉族自治县", "民和回族土族自治县", "民和回族土族自治县", "门源回族自治县", "海南藏族自治州", "河南蒙古族自治县"],
  "新疆": ["乌鲁木齐", "克孜勒苏柯尔克孜", "阿勒泰", "五家渠", "石河子", "伊犁", "吐鲁番", "塔城", "克拉玛依", "喀什", "和田", "哈密",
           "昌吉", "博尔塔拉", "阿克苏", "巴音郭楞", "阿拉尔", "图木舒克", "铁门关", "北屯", "双河", "可克达拉", "昆玉", "胡杨河",
           "新星", "白杨", "吉木乃县","和布克赛尔蒙古自治县", "塔什库尔干塔吉克自治县", "察布查尔锡伯自治县", "巴里坤哈萨克自治县",
           "木垒哈萨克自治县", "焉耆回族自治县"],
  "贵州": ["铜仁", "黔东南", "贵阳", "安顺", "遵义", "黔西南", "黔南", "六盘水", "毕节", "三都水族自治县", "关岭布依族苗族自治县",
           "务川仡佬族苗族自治县", "印江土家族苗族自治县", "威宁彝族回族苗族自治县", "松桃苗族自治县", "沿河土家族自治县", "玉屏侗族自治县",
           "紫云苗族布依族自治县", "道真仡佬族苗族自治县", "镇宁布依族苗族自治县"],
  "四川": ["遂宁", "攀枝花", "眉山", "成都", "巴中", "广安", "自贡", "资阳", "宜宾", "雅安", "内江", "南充", "绵阳", "泸州", "乐山",
           "广元", "德阳", "达州", "阿坝", "甘孜", "凉山", "峨边彝族自治县", "木里藏族自治县", "马边彝族自治县", "北川羌族自治县"],
  "上海": ["上海"],
  "广西": ["南宁", "贵港", "玉林", "梧州", "钦州", "柳州", "来宾", "贺州", "河池", "桂林", "防城港", "崇左", "北海", "百色",
           "三江侗族自治县", "大化瑶族自治县", "富川瑶族自治县", "巴马瑶族自治县", "恭城瑶族自治县", "罗城仫佬族自治县",
           "融水苗族自治县", "都安瑶族自治县", "金秀瑶族自治县", "隆林各族自治县", "龙胜各族自治县", "环江毛南族自治县"],
  "西藏": ["拉萨", "林芝", "昌都", "日喀则", "那曲", "阿里", "山南"],
  "云南": ["昆明", "玉溪", "昭通", "曲靖", "普洱", "临沧", "丽江", "保山", "红河", "大理", "西双版纳", "文山", "德宏", "怒江",
           "迪庆", "楚雄", "石林彝族自治县", "禄劝彝族苗族自治县", "巍山彝族回族自治县", "寻甸回族彝族自治县", "峨山彝族自治县",
           "新平彝族傣族自治县", "新平彝族傣族自治县", "元江哈尼族彝族傣族自治县", "玉龙纳西族自治县", "宁蒗彝族自治县",
           "宁洱哈尼族彝族自治县", "墨江哈尼族自治县", "景东彝族自治县", "景谷傣族彝族自治县", "镇沅彝族哈尼族拉祜族自治县",
           "江城哈尼族彝族自治县", "孟连傣族拉祜族佤族自治县", "澜沧拉祜族自治县", "西盟佤族自治县", "双江拉祜族佤族布朗族傣族自治县",
           "耿马傣族佤族自治县", "沧源佤族自治县", "屏边苗族自治县", "金平苗族瑶族傣族自治县", "河口瑶族自治县", "漾濞彝族自治县",
           "南涧彝族自治县", "巍山彝族回族自治县", "贡山独龙族怒族自治县", "兰坪白族普米族自治县", "维西傈僳族自治县"],
  "内蒙古": ["呼和浩特", "乌兰察布", "赤峰", "呼伦贝尔", "乌海", "通辽", "巴彦淖尔", "鄂尔多斯", "包头", "兴安", "锡林郭勒",
             "阿拉善", "莫力达瓦达斡尔族自治旗", "鄂伦春自治旗", "鄂温克族自治旗"],
  "海南": ["海口", "三沙", "三亚", "儋州", "临高", "五指山", "文昌", "万宁", "澄迈", "屯昌", "定安", "东方", "琼海", "白沙黎族自治县",
           "昌江黎族自治县", "乐东黎族自治县", "陵水黎族自治县", "保亭黎族苗族自治县", "琼中黎族苗族自治县"],
  "重庆": ["重庆", "彭水苗族土家族自治县", "石柱土家族自治县", "秀山土家族苗族自治县", "酉阳土家族苗族自治县"]
}
province_all = "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建", "江西",\
            "山东", "河南", "湖北", "湖南", "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", \
            "北京", "天津", "上海", "重庆", "内蒙古", "广西", "宁夏", "新疆", "西藏"

def city2province(city_list):
  out_result = {}
  for prov, cities in city_list.items():
    for city in cities:
      out_result[city] = prov
  return out_result
city2province = city2province(city_list)