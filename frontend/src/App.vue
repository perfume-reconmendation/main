<template>
  <div id="app">
    <div class="background">
      <div class="background-top"></div>
      <div class="background-bottom"></div>
    </div>
    <div class="page">
      <Header :current_page="current" />
    </div>
    <div class="wrap">
      <Menu :img_src="getSrc()" />
      <div class="form">
        <form action="/result" method="post">
          <Content
            :current="current"
            :current_page="getCurrentForm()"
            @answerUpdate="answerUpdate"
          />
          <div class="buttons">
            <button type="button" value="-1" @click="pageUpdate">이전</button>
            <button type="button" value="1" @click="pageUpdate">다음</button>
          </div>
          <input type="hidden" name="data" :value="output" />
          <button type="submit">result</button>
        </form>
      </div>
    </div>
  </div>
</template>

<script>
import Header from "./components/Header";
import Content from "./components/Content";
import Menu from "./components/Menu";

export default {
  name: "App",
  components: {
    Header,
    Content,
    Menu,
  },
  data() {
    return {
      current: 0,
      output: "",
      qs: [
        {
          src: "https://picsum.photos/400/800?random=1",
          q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?",
          answer: "",
        },
        {
          src: "https://picsum.photos/400/800?random=2",
          q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?2",
          answer: "",
        },
        {
          src: "https://picsum.photos/400/800?random=3",
          q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?3",
          answer: "",
        },
        {
          src: "https://picsum.photos/400/800?random=4",
          q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?4",
          answer: "",
        },
        {
          src: "https://picsum.photos/400/800?random=5",
          q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?5",
          answer: "",
        },
        // {
        //   src: "https://picsum.photos/400/800?random=6",
        //   q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?6",
        //   answer: "",
        // },
        // {
        //   src: "https://picsum.photos/400/800?random=7",
        //   q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?7",
        //   answer: "",
        // },
        // {
        //   src: "https://picsum.photos/400/800?random=8",
        //   q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?8",
        //   answer: "",
        // },
        // {
        //   src: "https://picsum.photos/400/800?random=9",
        //   q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?9",
        //   answer: "",
        // },
        // {
        //   src: "https://picsum.photos/400/800?random=10",
        //   q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?10",
        //   answer: "",
        // },
        // {
        //   src: "https://picsum.photos/400/800?random=11",
        //   q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?11",
        //   answer: "",
        // },
        // {
        //   src: "https://picsum.photos/400/800?random=12",
        //   q: "당신이 좋아하는 색상과 그 색상이 들어간 옷은?12",
        //   answer: "",
        // },
      ],
    };
  },
  methods: {
    getCurrentForm() {
      return this.qs[this.current];
    },
    getSrc() {
      return this.qs[this.current].src;
    },
    answerUpdate(index, text) {
      const qs = [...this.qs];
      const q = qs[index];
      // console.log(index, text);
      if (q) {
        q.answer = text;
        this.qs = qs;
      }
      var answer = "";
      this.qs.forEach((element) => {
        answer += element.answer + " ";
      });
      this.output = answer;
      // for (const q of qs) {
      //   console.log(q.q);
      //   console.log(q.answer);
      // }
    },
    pageUpdate(e) {
      this.current += Number(e.target.value);
      console.log(this.current);
    },
  },
};
</script>
<style scoped>
.background {
  width: 100vw;
  height: 100vh;
  z-index: -1;
  position: fixed;
}
.background-top {
  height: 41%;
  width: 100%;
  background-color: white;
}
.background-bottom {
  height: 59%;
  width: 100%;
  background-color: #ffece5;
}
.page {
  width: 100%;
  height: 150px;
  line-height: 150px;
  text-align: center;
}
.wrap {
  width: 100%;
  height: 600px;
  display: flex;
}
.form {
  flex: 1;
  text-align: center;
}
</style>
