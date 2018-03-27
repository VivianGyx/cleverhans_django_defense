// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('echarts'));

var posList = [
    'left', 'right', 'top', 'bottom',
    'inside',
    'insideTop', 'insideLeft', 'insideRight', 'insideBottom',
    'insideTopLeft', 'insideTopRight', 'insideBottomLeft', 'insideBottomRight'
];

myChart.configParameters = {
    rotate: {
        min: -90,
        max: 90
    },
    align: {
        options: {
            left: 'left',
            center: 'center',
            right: 'right'
        }
    },
    verticalAlign: {
        options: {
            top: 'top',
            middle: 'middle',
            bottom: 'bottom'
        }
    },
    position: {
        options: echarts.util.reduce(posList, function (map, pos) {
            map[pos] = pos;
            return map;
        }, {})
    },
    distance: {
        min: 0,
        max: 100
    }
};

myChart.config = {
    rotate: 90,
    align: 'left',
    verticalAlign: 'middle',
    position: 'insideBottom',
    distance: 15,
    onChange: function () {
        var labelOption = {
            normal: {
                rotate: myChart.config.rotate,
                align: myChart.config.align,
                verticalAlign: myChart.config.verticalAlign,
                position: myChart.config.position,
                distance: myChart.config.distance
            }
        };
        myChart.setOption({
            series: [{
                label: labelOption
            }, {
                label: labelOption
            }, {
                label: labelOption
            }, {
                label: labelOption
            }]
        });
    }
};

var labelOption = {
    normal: {
        show: true,
        position: myChart.config.position,
        distance: myChart.config.distance,
        align: myChart.config.align,
        verticalAlign: myChart.config.verticalAlign,
        rotate: myChart.config.rotate,
        formatter: '{c}  {name|{a}}',
        fontSize: 16,
        rich: {
            name: {
                textBorderColor: '#fff'
            }
        }
    }
};

var option = {
    // color: ['#003366', '#006699', '#4cabce'],
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        }
    },
    legend: {
        data: ['clean', 'fgsm', 'pgd']
    },
    toolbox: {
        show: true,
        orient: 'vertical',
        left: 'right',
        top: 'center',
        feature: {
            mark: {show: true},
            dataView: {show: true, readOnly: false},
            magicType: {show: true, type: ['line', 'bar', 'stack', 'tiled']},
            restore: {show: true},
            saveAsImage: {show: true}
        }
    },
    calculable: true,
    xAxis: [
        {
            type: 'category',
            axisTick: {show: false},
            data: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        }
    ],
    yAxis: [
        {
            type: 'value'
        }
    ],
    series: [
        {
            name: 'clean',
            type: 'bar',
            barGap: 0,
            label: labelOption
        },
        {
            name: 'fgsm',
            type: 'bar',
            label: labelOption
        },
        {
            name: 'pgd',
            type: 'bar',
            label: labelOption
        }
    ]
};

//加载数据
$(function () {
    var main = new Main();
    $('#recognizeDraw').click(function () {
        loadDATA(option);
        myChart.setOption(option);

    });
});

function loadDATA(option){
    $.ajax({
        type : "post",
        async : false, //同步执行
        url : "process",//路径配置
        data : {"inputs": JSON.stringify(inputs)},
        dataType : "json", //返回数据形式为json
        success : function(result) {
            if (result) {
                //初始化option.xAxis[0]中的data
                // option.xAxis.data=[];
                // for(var i=0;i<result.length;i++) {
                //     option.xAxis.data.push(result[i].name);
                // }
                //初始化option.series[0]中的data
                var newData = eval(result);
                for (var _i = 0; _i < 3; _i++) {
                    option.series[_i].data=[];
                    for(var _j = 0; _j < 10; _j++) {
                        var value = Math.round(newData[_i][_j]*1000) / 1000;
                        option.series[_i].data.push(value);
                    }
                }
            }
        }
    });
}