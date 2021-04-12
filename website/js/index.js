let app = new Vue({
    el: '#app',
    data: {
        navLinks: [
            {
                name: 'GitHub',
                url: 'https://github.com/awslabs/djl'
            },
            {
                name: 'Documentation',
                url: 'https://docs.djl.ai/'
            },
            {
                name: 'JavaDoc',
                url: 'website/javadoc.html'
            },
            {
                name: 'Demos',
                url: 'website/demo.html'
            },
            {
                name: 'Blogs',
                url: 'website/blog.html'
            },
            {
                name: 'Tutorial',
                url: 'https://docs.djl.ai/jupyter/tutorial/index.html'
            },
            {
                name: 'Examples',
                url: 'https://github.com/awslabs/djl/tree/master/examples'
            },
            {
                name: 'Slack',
                url: 'https://deepjavalibrary.slack.com/join/shared_invite/zt-ar91gjkz-qbXhr1l~LFGEIEeGBibT7w#/'
            },
            {
                name: 'D2L-Java Book',
                url: 'https://d2l.djl.ai'
            }
        ],
        page1: {
            title: 'Deep Java Library',
            abstract: 'Open source library to build and deploy deep learning in Java',
            getStartUtl: 'https://docs.djl.ai/docs/quick_start.html',
            gitHubUrl: 'https://github.com/awslabs/djl',
        },
        page2: {
            feature: [
                {
                    title: 'Engine Agnostic',
                    des: 'Write once and run anywhere. Develop your model using DJL and run it on an engine of your choice',
                    icon: 'device_hub'
                },
                {
                    title: 'Built for Java developers',
                    des: 'Intuitive APIs use native Java concepts and abstract away complexity involved with Deep learning',
                    icon: 'group'
                },
                {
                    title: 'Ease of deployment',
                    des: 'Bring in your own model, or use a state of the art model from our library to deploy in minutes',
                    icon: 'cloud_upload'
                }
            ],
        },
        page4: {
            title: 'From our customers',
            customers: [
                {
                    by: '-- Stanislav Kirdey, Engineer at Netflix observability team',
                    words: `“The Netflix observability team's future plans with DJL include trying out its training API, scaling usage of transfer learning inference,and exploring its bindings for PyTorch and MXNet to harness the power and availability of transfer learning.”`
                }, {
                    by: `-- Xiaoyan Zhang, Data Scientist at TalkingData`,
                    words: `“Using DJL allowed us to run large batch inference on Spark for Pytorch models. DJL helped reduce inference time from over six hours to under two hours.”`
                }, {
                    by: `-- Hermann Burgmeier, Engineer at Amazon Advertising team `,
                    words: `“DJL enables us to run models built with different ML frameworks side by side in the same JVM without infrastructure changes. ”`
                }, {
                    by: '-- Vaibhav Goel, Engineer at Amazon Behavior Analytics team',
                    words: `“Our science team prefers using Python. Our engineering team prefers using Java/Scala. With DJL, data science team can build models in different Python APIs such as Tensorflow, Pytorch, and MXNet, and engineering team can run inference on these models using DJL. We found that our  batch inference time was reduced by 85% from using DJL.”`
                }
            ]
        },
        page5: {
            title: 'About Us',
            abstract: 'Built with ❤️ for the Java community by',
            contactUrl: 'https://docs.djl.ai/leaders.html'
        }
    },
    created() {

    },
    mounted() {
        var mySwiper = new Swiper('#app.swiper-container', {
            direction: 'vertical',
            loop: false,
            // effect: 'cube',
            keyboardControl: true,
            mousewheelControl: true,

            // 如果需要分页器
            pagination: '.swiper-pagination',
            paginationClickable: true,

            // 如果需要前进后退按钮
            // nextButton: '.swiper-button-next',
            // prevButton: '.swiper-button-prev',

            // 如果需要滚动条
            // scrollbar: '.swiper-scrollbar',

            onTransitionStart: function (swiper) {
                // debugger
            }
        })
        var mySwiper2 = new Swiper('.customerWrap .swiper-container', {
            direction: 'horizontal',
            loop: true,
            effect: 'cube',
            speed: 1000,
            keyboardControl: true,
            mousewheelControl: true,

            // 如果需要分页器
            pagination: '.customer-swiper-pagination',
            paginationClickable: true,

            // 如果需要前进后退按钮
            // nextButton: '.swiper-button-next',
            // prevButton: '.swiper-button-prev',

            // 如果需要滚动条
            // scrollbar: '.swiper-scrollbar',

        })

    },
    methods: {

    }
})