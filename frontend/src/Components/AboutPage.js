import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography'


const AboutPage = () => {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container>
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
          <Box sx={{
            flexGrow: 1,
            backgroundColor: 'primary.light',
            padding: 10
          }}>
            <Typography
              variant='h4'
              noWrap
              component="div">
              Detailed Project Description
            </Typography>
          </Box> 
        </Grid>
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
          <Box sx={{
            flexGrow: 1,
            backgroundColor: 'primary.light',
            padding: 10
          }}>
            <Typography
              variant='h4'
              noWrap
              component="div">
              Contributors
            </Typography>
          </Box> 
        </Grid>
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
          <Box sx={{
            flexGrow: 1,
            backgroundColor: 'primary.light',
            padding: 10
          }}>
            <Typography
              variant='h4'
              noWrap
              component="div">
              Links
            </Typography>
          </Box> 
        </Grid>
      </Grid>
    </Box>
  )
}

export default AboutPage