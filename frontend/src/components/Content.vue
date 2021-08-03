<template>
  <div class="question-form">
    <p>Q{{ current + 1 }}</p>
    <p>{{ current_page.q }}</p>
    <textarea
      name
      id
      cols="30"
      placeholder="I like white and I have white shirts that I bought on a rainy day."
      v-model="text"
      @change="handleUpdate"
      rows="10"
    ></textarea>
  </div>
</template>

<script>
export default {
  props: ["current_page", "current"],
  data() {
    return {
      // current: this.current,
      text: this.current_page.answer,
    };
  },
  methods: {
    handleUpdate() {
      this.$emit("answerUpdate", this.current, this.text);
    },
  },
  watch: {
    current_page: function (val) {
      // console.log(val);
      this.text = val.answer;
    },
  },
};
</script>

<style scoped>
.question-form {
}
</style>
