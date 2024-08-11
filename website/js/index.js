let app = new Vue({
    el: '#app',
    data: {
        hasRenderTwitter: false,
        customerWrap_page4: '',
        footerSwiper_page5: '',
        navLinks: [
            {
                name: 'GitHub',
                url: 'https://github.com/deepjavalibrary/djl'
            },
            {
                name: 'Documentation',
                url: 'https://docs.djl.ai/'
            },
            {
                name: 'JavaDoc',
                url: 'https://djl.ai/website/javadoc.html'
            },
            {
                name: 'Demos',
                url: 'https://github.com/deepjavalibrary/djl-demo?tab=readme-ov-file#deep-java-library-examples'
            },
            {
                name: 'Blogs',
                url: 'https://djl.ai/website/blog.html'
            },
            {
                name: 'Tutorial',
                url: 'https://docs.djl.ai/master/docs/demos/jupyter/tutorial/index.html'
            },
            {
                name: 'Examples',
                url: 'https://github.com/deepjavalibrary/djl/tree/master/examples'
            },
            {
                name: 'Slack',
                url: 'https://deepjavalibrary.slack.com/join/shared_invite/zt-ar91gjkz-qbXhr1l~LFGEIEeGBibT7w#/'
            },
            {
                name: 'D2L-Java Book',
                url: 'https://d2l.djl.ai'
            },
            {
                name: 'version1.0',
                url: 'https://djl.ai/index1.0.html'
            }
        ],
        page1: {
            title: 'Deep Java Library',
            abstract: 'Open source library to build and deploy deep learning in Java',
            getStartUtl: 'https://docs.djl.ai/master/docs/quick_start.html',
            gitHubUrl: 'https://github.com/deepjavalibrary/djl',
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
            contactUrl: 'https://docs.djl.ai/master/leaders.html',
            leadrs: [
                {
                    name: 'anthony',
                    url: './website/img/community_leader/anthony.jpeg'
                },
                {
                    name: 'christoph',
                    url: './website/img/community_leader/christoph.png',
                },
                {
                    name: 'erik',
                    url: './website/img/community_leader/erik.png',
                }, {
                    name: 'frank',
                    url: './website/img/community_leader/frank.png',
                }, {
                    name: 'jake',
                    url: './website/img/community_leader/jake.jpg',
                }, {
                    name: 'keerthan',
                    url: './website/img/community_leader/keerthan.jpg',
                }, {
                    name: 'kimi',
                    url: './website/img/community_leader/kimi.png',
                }, {
                    name: 'lai',
                    url: './website/img/community_leader/lai.jpg',
                }, {
                    name: 'lu',
                    url: './website/img/community_leader/lu.png',
                }, {
                    name: 'qing',
                    url: './website/img/community_leader/qing.jpeg',
                }, {
                    name: 'stanislav',
                    url: './website/img/community_leader/stanislav.png',
                }, {
                    name: 'wei',
                    url: './website/img/community_leader/wei.jpg',
                }, {
                    name: 'zach',
                    url: './website/img/community_leader/zach.jpeg',
                }, {
                    name: 'fyz',
                    url: './website/img/community_leader/fyz.jpg',
                }
            ]
        }
    },
    created() {

    },
    mounted() {
        $('.sidenav').sidenav();
        if (typeof Swiper === 'undefined') return
        var mySwiper = new Swiper('.app-swiper-container', {
            direction: 'vertical',
            loop: false,
            // effect: 'cube',
            keyboardControl: true,
            mousewheelControl: true,
            longSwipesRatio: 0.1,
            // 如果需要分页器
            pagination: '.swiper-pagination',
            paginationClickable: true,
            shortSwipes: true,

            noSwiping: true,
            lazyLoading: true,

            onSlideChangeStart: (swiper) => {
                // console.log(swiper, swiper.activeIndex);
                if (swiper.activeIndex === 2) {
                    if (this.hasRenderTwitter) return
                    let srciptEle = document.createElement('script')
                    srciptEle.src = 'https://platform.twitter.com/widgets.js'
                    document.body.appendChild(srciptEle)
                    srciptEle.onload = () => {
                        this.hasRenderTwitter = true
                    }

                } else if (swiper.activeIndex === 3) {
                    if (this.customerWrap_page4) return
                    this.customerWrap_page4 = new Swiper('.customerWrap .swiper-container', {
                        direction: 'horizontal',
                        loop: false,
                        autoplay: 3000,
                        autoplayStopOnLast: true,
                        parallax: true,
                        // effect: 'cube',
                        speed: 2000,
                        keyboardControl: true,
                        mousewheelControl: true,
                        lazyLoading: true,

                        // 如果需要分页器
                        pagination: '.customer-swiper-pagination',
                        paginationClickable: true,

                        // 如果需要前进后退按钮
                        nextButton: '.swiper-button-next',
                        prevButton: '.swiper-button-prev',

                        // 如果需要滚动条
                        // scrollbar: '.swiper-scrollbar',

                    })
                } else if (swiper.activeIndex === 4) {
                    if (this.footerSwiper_page5) return
                    this.footerSwiper_page5 = new Swiper('.app-footer .swiper-container', {
                        direction: 'horizontal',
                        loop: true,
                        autoplay: 3000,
                        autoplayDisableOnInteraction: false,
                        // grabCursor: true,
                        // effect: 'cube',
                        // effect: 'coverflow',
                        slidesPerView: 4,
                        spaceBetween: 5,
                        shortSwipes: true,

                        speed: 1000,
                        keyboardControl: true,
                        mousewheelControl: true,
                        lazyLoading: true,

                        // 如果需要分页器
                        paginationType: 'progress',
                        pagination: '.leader-swiper-pagination',
                        paginationClickable: true,

                    })
                }
            }

        })


    },
    methods: {

    }
})
