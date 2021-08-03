<template>
  <div>
    <img :src="img_src" alt="" />
  </div>
</template>
<script>
export default {
  props: {
    img_src: { type: String, default: "" }
  }
};
</script>
<style scoped>
div {
  flex: 1;
  text-align: center;
}
img {
  width: 400px;
  height: 100%;
}
</style>
